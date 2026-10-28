/* Host test for mic_features.c. Build and run on the PC:
   gcc -Wall -Wextra -Werror -O2 host_test.c mic_features.c -lm -o host_test && ./host_test */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mic_features.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint16_t buf[2000];

int main(void)
{
    mic_feat_t f;
    uint32_t i, k;

    /* square root is exact */
    CHECK(mic_isqrt64(0) == 0 && mic_isqrt64(1) == 1 && mic_isqrt64(3) == 1 && mic_isqrt64(4) == 2, "small sqrt");
    for (k = 0; k < 70000; k++) {
        uint64_t sq = (uint64_t)k * k;
        CHECK(mic_isqrt64(sq) == k, "sqrt(%llu)", (unsigned long long)sq);
        CHECK(mic_isqrt64(sq + (k ? k * 2u : 0u)) == k, "sqrt just below next square, k=%u", k);
    }

    /* empty window: all zeros, no division by zero */
    mic_feat_compute(buf, 0, &f);
    CHECK(f.n == 0 && f.mean == 0 && f.rms == 0 && f.peak == 0 && f.zcr == 0 && f.hf == 0 && f.clip == 0, "empty");

    /* one sample */
    buf[0] = 8000; mic_feat_compute(buf, 1, &f);
    CHECK(f.mean == 8000 && f.rms == 0 && f.peak == 0 && f.zcr == 0 && f.hf == 0, "one sample");

    /* constant signal: everything about movement is zero */
    for (i = 0; i < 2000; i++) buf[i] = 8440;
    mic_feat_compute(buf, 2000, &f);
    CHECK(f.mean == 8440 && f.rms == 0 && f.peak == 0 && f.zcr == 0 && f.hf == 0 && f.clip == 0, "constant");

    /* alternating +-100 around 8000: rms 100, peak 100, crosses at every step, hf 200 */
    for (i = 0; i < 2000; i++) buf[i] = (i & 1) ? 8100 : 7900;
    mic_feat_compute(buf, 2000, &f);
    CHECK(f.mean == 8000 && f.rms == 100 && f.peak == 100 && f.zcr == 1999 && f.hf == 200, "alternating: mean %u rms %u peak %u zcr %u hf %u",
          f.mean, f.rms, f.peak, f.zcr, f.hf);

    /* full scale square wave: every sample is clipped */
    for (i = 0; i < 2000; i++) buf[i] = (i & 1) ? 16383 : 0;
    mic_feat_compute(buf, 2000, &f);
    CHECK(f.clip == 2000 && f.peak >= 8191, "full scale: clip %u peak %u", f.clip, f.peak);

    /* a single tap spike in a flat window: peak is large, rms small, zcr small */
    for (i = 0; i < 2000; i++) buf[i] = 8440;
    buf[1000] = 16000;
    mic_feat_compute(buf, 2000, &f);
    CHECK(f.peak > 7000 && f.rms < 200 && f.zcr <= 2 && f.clip == 0, "spike: peak %u rms %u zcr %u clip %u", f.peak, f.rms, f.zcr, f.clip);

    /* exactly one clipped sample at each end value */
    for (i = 0; i < 2000; i++) buf[i] = 8440;
    buf[10] = 16383; buf[20] = 0; buf[30] = 16382; buf[40] = 1;
    mic_feat_compute(buf, 2000, &f);
    CHECK(f.clip == 2, "clip counts only 0 and 16383: %u", f.clip);

    /* a slow ramp crosses the mean exactly once */
    for (i = 0; i < 2000; i++) buf[i] = (uint16_t)(7000 + i);
    mic_feat_compute(buf, 2000, &f);
    CHECK(f.zcr == 1 && f.hf == 1, "ramp: zcr %u hf %u", f.zcr, f.hf);

    /* 300 random windows of random length against a double-precision reference */
    srand(12345);
    for (k = 0; k < 300; k++) {
        uint32_t n = 2 + (uint32_t)(rand() % 1999);
        uint32_t spread = 1 + (uint32_t)(rand() % 8000);
        uint32_t centre = 4000 + (uint32_t)(rand() % 8000);
        double s = 0, ss = 0, dd = 0; long m; unsigned zc = 0, cl = 0, pk = 0; int pa = 0;
        for (i = 0; i < n; i++) {
            long v = (long)centre + (long)(rand() % (2 * spread + 1)) - (long)spread;
            if (v < 0) v = 0;
            if (v > 16383) v = 16383;
            buf[i] = (uint16_t)v;
        }
        for (i = 0; i < n; i++) { s += buf[i]; if (buf[i] == 0 || buf[i] == 16383) cl++; }
        m = (long)floor(s / n + 0.5);
        for (i = 0; i < n; i++) {
            double d = (double)buf[i] - (double)m; int above = d >= 0;
            ss += d * d;
            if (fabs(d) > pk) pk = (unsigned)fabs(d);
            if (i > 0) { double st = (double)buf[i] - buf[i - 1]; dd += st * st; if (above != pa) zc++; }
            pa = above;
        }
        mic_feat_compute(buf, n, &f);
        CHECK(f.mean == (uint32_t)m, "random %u: mean %u vs %ld", k, f.mean, m);
        CHECK(f.rms == (uint32_t)floor(sqrt(ss / n)), "random %u: rms %u vs %.3f", k, f.rms, sqrt(ss / n));
        CHECK(f.hf == (uint32_t)floor(sqrt(dd / (n - 1))), "random %u: hf %u vs %.3f", k, f.hf, sqrt(dd / (n - 1)));
        CHECK(f.peak == pk && f.zcr == zc && f.clip == cl, "random %u: peak %u/%u zcr %u/%u clip %u/%u", k, f.peak, pk, f.zcr, zc, f.clip, cl);
    }

    if (failures) { printf("%d check(s) FAILED\n", failures); return 1; }
    printf("all checks passed\n");
    return 0;
}
