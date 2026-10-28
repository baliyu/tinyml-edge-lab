#include "mic_stats.h"

void mic_win_reset(mic_win_t *w)
{
    w->n = 0;
    w->min = 0xFFFFFFFFu;
    w->max = 0;
    w->sum = 0;
    w->sumsq = 0;
}

void mic_win_add(mic_win_t *w, uint32_t sample)
{
    if (sample < w->min) w->min = sample;
    if (sample > w->max) w->max = sample;
    w->sum += sample;
    w->sumsq += (uint64_t)sample * sample;
    w->n++;
}

uint32_t mic_isqrt(uint64_t v)
{
    uint64_t lo = 0, hi = 0xFFFFFFFFu;   /* sqrt of a 64-bit value fits in 32 bits */
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1) / 2;
        if (mid * mid <= v) lo = mid; else hi = mid - 1;
    }
    return (uint32_t)lo;
}

void mic_win_summary(const mic_win_t *w, mic_summary_t *out)
{
    out->n = w->n;
    if (w->n == 0) {
        out->min = out->max = out->mean = out->rms = 0;
        return;
    }
    out->min = w->min;
    out->max = w->max;
    out->mean = (uint32_t)(w->sum / w->n);
    /* variance = (sum of squares - sum*sum/n) / n */
    uint64_t var = (w->sumsq - (w->sum * w->sum) / w->n) / w->n;
    out->rms = mic_isqrt(var);
}
