/* The whole on-device chain for the desk-gesture classifier, in plain C99 with no malloc and no hardware access:
 *
 *   raw samples (a 32-bit tick count and x/y/z in milli-g, at the sensor's own, slightly uneven rate)
 *     -> linear resampling onto an exact 50 Hz grid -> 128-sample windows, a new one every 64 samples
 *     -> window features (edge_features.c) -> the trained model (gesture_model.c) -> a gesture number.
 *
 * It does the same job as tinyml_lab.edge_gestures (resample_to_grid, make_windows, feature_matrix) followed by the model's
 * predict(). tests/test_gesture_pipeline_c.py and scripts/check_gesture_pipeline.py hold it to that on the PC, so the board
 * only has to call gp_push() once per sensor sample.
 *
 * Time is kept as an integer in units of 1/(50 * tick_hz) second. In those units one grid step is exactly tick_hz, so the
 * grid does not drift or round, whatever the tick rate (3 MHz internal oscillator or 32.768 kHz crystal).
 *
 * Usage:
 *     static gp_state g;
 *     gp_init(&g, tick_hz, 0);
 *     ... for every sensor sample:  if (gp_push(&g, ticks, x_mg, y_mg, z_mg)) use GESTURE_CLASS_NAMES[gp_class(&g)];
 */
#ifndef GESTURE_PIPELINE_H
#define GESTURE_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "edge_features.h"
#include "gesture_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GP_FS_HZ 50                 /* the grid every window is cut from; the features and the model assume it */
#define GP_WINDOW EF_WINDOW         /* samples per window (2.56 s) */
#define GP_STEP 64                  /* a new window every 64 samples: 50 % overlap */
#define GP_MAX_GAP_STEPS 12         /* a hole longer than 12 grid steps (0.24 s) is not interpolated over: the window restarts */

typedef struct {
    uint32_t tick_hz;               /* ticks per second of the timer that stamps the samples */
    uint32_t skip;                  /* grid samples to drop after start-up, before the first window */
    uint32_t last_raw;              /* the last 32-bit tick count seen */
    uint64_t u_prev;                /* time of the previous sample, in units of 1/(50 * tick_hz) s since the first */
    uint32_t grid_k;                /* index of the next grid sample to produce */
    uint32_t to_skip;               /* start-up samples still to drop */
    uint32_t fill;                  /* samples in the window being built */
    uint32_t windows;               /* windows classified so far */
    uint32_t restarts;              /* times a long gap restarted the window */
    uint32_t ignored;               /* samples dropped because they repeated the previous timestamp */
    int32_t prev[3];                /* the previous sample, milli-g */
    int last_class;                 /* the last window's gesture number, or -1 */
    int started;                    /* has a first sample defined time zero? */
    ef_real buf[GP_WINDOW * 3];     /* the window being built, x/y/z interleaved */
    ef_real feat[GESTURE_N_FEATURES];
} gp_state;

/* Returns 0, or -1 for a null state or a tick rate outside 1 .. 100 000 000. skip: grid samples dropped after start-up. */
int gp_init(gp_state *g, uint32_t tick_hz, uint32_t skip);

/* One sensor sample. ticks is the free-running 32-bit counter (it may wrap); x, y, z are milli-g.
 * Returns 1 when this sample completed a window and gp_class() holds that window's gesture, otherwise 0. */
int gp_push(gp_state *g, uint32_t ticks, int32_t x_mg, int32_t y_mg, int32_t z_mg);

int gp_class(const gp_state *g);                /* the last window's gesture number, an index into GESTURE_CLASS_NAMES, or -1 */
uint32_t gp_windows(const gp_state *g);
uint32_t gp_restarts(const gp_state *g);
uint32_t gp_ignored(const gp_state *g);
#ifdef GP_TEST_HOOKS                            /* PC tests only: kept out of the board build (the double return pulls in a libgcc helper) */
double gp_feature(const gp_state *g, int i);    /* feature i of the last window */
size_t gp_state_size(void);                     /* sizeof(gp_state) */
#endif

#ifdef __cplusplus
}
#endif
#endif
