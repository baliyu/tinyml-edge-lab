/* Per-window summary numbers for the SparkFun Edge microphone (pure C, no SDK, host-tested).
   One window is WINDOW_N raw 14-bit ADC samples (0..16383). All maths is integer, so the board and the PC agree. */
#ifndef MIC_FEATURES_H
#define MIC_FEATURES_H
#include <stdint.h>

#define MIC_ADC_MAX 16383u

typedef struct {
    uint32_t n;      /* samples used */
    uint32_t mean;   /* average, rounded to the nearest count */
    uint32_t rms;    /* sqrt(mean of (x - mean)^2), rounded down: loudness */
    uint32_t peak;   /* largest |x - mean|: how big the biggest swing is */
    uint32_t zcr;    /* times the signal crosses the window mean (a value equal to the mean counts as above it) */
    uint32_t hf;     /* sqrt(mean of (x[i] - x[i-1])^2), rounded down: how fast the signal changes */
    uint32_t clip;   /* samples at 0 or at MIC_ADC_MAX */
} mic_feat_t;

/* Compute the summary of x[0..n-1]. n == 0 gives all zeros. */
void mic_feat_compute(const uint16_t *x, uint32_t n, mic_feat_t *out);

/* floor(sqrt(v)), exact */
uint32_t mic_isqrt64(uint64_t v);

#endif
