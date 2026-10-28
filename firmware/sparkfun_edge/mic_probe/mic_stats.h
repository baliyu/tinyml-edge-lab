/* mic_stats: one-second statistics of microphone samples. Pure C, no hardware, host-tested.
 *
 * Assumptions (checked by the host test): sample values are below 2^16 and a window holds at most 2^15 samples,
 * so the 64-bit sums cannot overflow. The board uses 14-bit samples and 8000 samples per window. */
#ifndef MIC_STATS_H
#define MIC_STATS_H

#include <stdint.h>

typedef struct {
    uint32_t n;
    uint32_t min;
    uint32_t max;
    uint64_t sum;
    uint64_t sumsq;
} mic_win_t;

typedef struct {
    uint32_t n;
    uint32_t min;
    uint32_t max;
    uint32_t mean;   /* floor of the average sample value */
    uint32_t rms;    /* root-mean-square distance from the mean (the standard deviation), a loudness measure */
} mic_summary_t;

void     mic_win_reset(mic_win_t *w);
void     mic_win_add(mic_win_t *w, uint32_t sample);
void     mic_win_summary(const mic_win_t *w, mic_summary_t *out);
uint32_t mic_isqrt(uint64_t v);   /* floor(sqrt(v)) */

#endif
