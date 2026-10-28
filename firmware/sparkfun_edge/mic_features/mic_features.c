#include "mic_features.h"

uint32_t mic_isqrt64(uint64_t v)
{
    uint64_t r = 0, bit = (uint64_t)1 << 62;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= r + bit) { v -= r + bit; r = (r >> 1) + bit; }
        else r >>= 1;
        bit >>= 2;
    }
    return (uint32_t)r;
}

void mic_feat_compute(const uint16_t *x, uint32_t n, mic_feat_t *out)
{
    uint64_t sum = 0, sumsq = 0, dsq = 0;
    uint32_t i, peak = 0, zcr = 0, clip = 0;
    int32_t m;
    int prev_above = 0;

    out->n = n; out->mean = 0; out->rms = 0; out->peak = 0; out->zcr = 0; out->hf = 0; out->clip = 0;
    if (n == 0) return;

    for (i = 0; i < n; i++) {
        sum += x[i];
        if (x[i] == 0 || x[i] >= MIC_ADC_MAX) clip++;
    }
    m = (int32_t)((sum + n / 2) / n);

    for (i = 0; i < n; i++) {
        int32_t d = (int32_t)x[i] - m;
        uint32_t a = (uint32_t)(d < 0 ? -d : d);
        int above = (d >= 0);
        sumsq += (uint64_t)a * a;
        if (a > peak) peak = a;
        if (i > 0) {
            int32_t step = (int32_t)x[i] - (int32_t)x[i - 1];
            uint32_t s = (uint32_t)(step < 0 ? -step : step);
            dsq += (uint64_t)s * s;
            if (above != prev_above) zcr++;
        }
        prev_above = above;
    }

    out->mean = (uint32_t)m;
    out->rms = mic_isqrt64(sumsq / n);
    out->peak = peak;
    out->zcr = zcr;
    out->hf = (n > 1) ? mic_isqrt64(dsq / (n - 1)) : 0;
    out->clip = clip;
}
