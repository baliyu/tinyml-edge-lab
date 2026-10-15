/* See gesture_pipeline.h. */
#include "gesture_pipeline.h"

#if GESTURE_USES_MOTION_FEATURES
#define GP_FEATURES ef_motion_features
#define GP_EXPECTED_FEATURES EF_MOTION_FEATURES
#else
#define GP_FEATURES ef_raw_features
#define GP_EXPECTED_FEATURES EF_RAW_FEATURES
#endif

_Static_assert(GESTURE_N_FEATURES == GP_EXPECTED_FEATURES, "the model and the feature code disagree about the number of features");
_Static_assert(GP_STEP > GP_MAX_GAP_STEPS, "one sensor sample must not be able to finish two windows");
_Static_assert(GP_STEP < GP_WINDOW, "windows must overlap");

/* One grid sample into the window; when the window is full, classify it and slide on by GP_STEP samples. */
static int gp_emit(gp_state *g, ef_real x, ef_real y, ef_real z)
{
    float model_in[GESTURE_N_FEATURES];
    ef_real *w;
    uint32_t i;

    if (g->to_skip > 0u) {
        g->to_skip--;
        return 0;
    }
    w = &g->buf[3u * g->fill];
    w[0] = x;
    w[1] = y;
    w[2] = z;
    g->fill++;
    if (g->fill < GP_WINDOW) {
        return 0;
    }
    GP_FEATURES(g->buf, g->feat);
    for (i = 0u; i < GESTURE_N_FEATURES; ++i) {
        model_in[i] = (float)g->feat[i];        /* the model's thresholds are float32; scikit-learn also casts its input to float32 */
    }
    g->last_class = gesture_model_predict(model_in);
    g->windows++;
    for (i = 0u; i < 3u * (GP_WINDOW - GP_STEP); ++i) {
        g->buf[i] = g->buf[3u * GP_STEP + i];
    }
    g->fill = GP_WINDOW - GP_STEP;
    return 1;
}

static void gp_restart(gp_state *g)
{
    g->started = 0;
    g->fill = 0u;
    g->grid_k = 0u;
}

int gp_init(gp_state *g, uint32_t tick_hz, uint32_t skip)
{
    if (g == NULL || tick_hz == 0u || tick_hz > 100000000u) {
        return -1;
    }
    g->tick_hz = tick_hz;
    g->skip = skip;
    g->to_skip = skip;
    g->windows = 0u;
    g->restarts = 0u;
    g->ignored = 0u;
    g->last_class = -1;
    gp_restart(g);
    return 0;
}

int gp_push(gp_state *g, uint32_t ticks, int32_t x_mg, int32_t y_mg, int32_t z_mg)
{
    uint64_t u_cur, span, gt;
    uint32_t dticks;
    int done = 0;

    if (g->started) {
        dticks = ticks - g->last_raw;                       /* unsigned arithmetic: right across the wrap of the 32-bit counter */
        if (dticks == 0u) {
            g->ignored++;                                   /* a repeated timestamp cannot be interpolated: keep the first sample */
            return 0;
        }
        if ((uint64_t)dticks * GP_FS_HZ > (uint64_t)GP_MAX_GAP_STEPS * g->tick_hz) {
            g->restarts++;                                  /* too long a hole (or time running backwards): start a fresh window */
            gp_restart(g);
        }
    }
    if (!g->started) {                                      /* this sample defines time zero, and is grid sample 0 */
        g->started = 1;
        g->last_raw = ticks;
        g->u_prev = 0u;
        g->prev[0] = x_mg;
        g->prev[1] = y_mg;
        g->prev[2] = z_mg;
        g->grid_k = 1u;
        return gp_emit(g, (ef_real)x_mg, (ef_real)y_mg, (ef_real)z_mg);
    }

    u_cur = g->u_prev + (uint64_t)dticks * GP_FS_HZ;
    span = u_cur - g->u_prev;
    for (;;) {                                              /* every grid sample that now lies between the previous sample and this one */
        ef_real f;
        gt = (uint64_t)g->grid_k * g->tick_hz;
        if (gt > u_cur) {
            break;
        }
        f = (ef_real)(uint32_t)(gt - g->u_prev) / (ef_real)(uint32_t)span;
        done |= gp_emit(g,
                        (ef_real)g->prev[0] + (ef_real)(x_mg - g->prev[0]) * f,
                        (ef_real)g->prev[1] + (ef_real)(y_mg - g->prev[1]) * f,
                        (ef_real)g->prev[2] + (ef_real)(z_mg - g->prev[2]) * f);
        g->grid_k++;
    }
    g->last_raw = ticks;
    g->u_prev = u_cur;
    g->prev[0] = x_mg;
    g->prev[1] = y_mg;
    g->prev[2] = z_mg;
    return done;
}

int gp_class(const gp_state *g) { return g->last_class; }
uint32_t gp_windows(const gp_state *g) { return g->windows; }
uint32_t gp_restarts(const gp_state *g) { return g->restarts; }
uint32_t gp_ignored(const gp_state *g) { return g->ignored; }
#ifdef GP_TEST_HOOKS
double gp_feature(const gp_state *g, int i) { return (double)g->feat[i]; }
size_t gp_state_size(void) { return sizeof(gp_state); }
#endif
