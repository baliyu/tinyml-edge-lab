/* Host test for mic_stats.c. Build and run on the PC:  gcc -Wall -Wextra -Werror -O2 host_test.c mic_stats.c -lm -o host_test && ./host_test */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mic_stats.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void summarise(const uint32_t *s, uint32_t n, mic_summary_t *out)
{
    mic_win_t w;
    mic_win_reset(&w);
    for (uint32_t i = 0; i < n; i++) mic_win_add(&w, s[i]);
    mic_win_summary(&w, out);
}

int main(void)
{
    static uint32_t buf[8000];
    mic_summary_t r;

    /* empty window: no division by zero, all zeros */
    mic_win_t w; mic_win_reset(&w); mic_win_summary(&w, &r);
    CHECK(r.n == 0 && r.min == 0 && r.max == 0 && r.mean == 0 && r.rms == 0, "empty window");

    /* constant signal: loudness 0 */
    for (int i = 0; i < 8000; i++) buf[i] = 5000;
    summarise(buf, 8000, &r);
    CHECK(r.n == 8000 && r.min == 5000 && r.max == 5000 && r.mean == 5000 && r.rms == 0, "constant: %u %u %u %u %u", r.n, r.min, r.max, r.mean, r.rms);

    /* alternating 4000 / 6000: mean 5000, standard deviation exactly 1000 */
    for (int i = 0; i < 8000; i++) buf[i] = (i & 1) ? 6000 : 4000;
    summarise(buf, 8000, &r);
    CHECK(r.min == 4000 && r.max == 6000 && r.mean == 5000 && r.rms == 1000, "alternating: %u %u %u %u", r.min, r.max, r.mean, r.rms);

    /* full-scale 14-bit extremes do not overflow */
    for (int i = 0; i < 8000; i++) buf[i] = (i & 1) ? 16383 : 0;
    summarise(buf, 8000, &r);
    CHECK(r.min == 0 && r.max == 16383 && r.mean == 8191 && r.rms >= 8190 && r.rms <= 8192, "full scale: %u %u %u %u", r.min, r.max, r.mean, r.rms);

    /* a single sample */
    buf[0] = 1234;
    summarise(buf, 1, &r);
    CHECK(r.n == 1 && r.min == 1234 && r.max == 1234 && r.mean == 1234 && r.rms == 0, "single sample");

    /* integer square root: exhaustive for small values, perfect squares and neighbours for large ones */
    for (uint64_t v = 0; v < 70000; v++) {
        uint32_t s = mic_isqrt(v);
        CHECK((uint64_t)s * s <= v && (uint64_t)(s + 1) * (s + 1) > v, "isqrt(%llu)=%u", (unsigned long long)v, s);
    }
    uint64_t big[] = { 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFE00000001ull, 0xFFFFFFFE00000000ull, 1ull << 62, (1ull << 62) - 1 };
    for (unsigned i = 0; i < sizeof big / sizeof big[0]; i++) {
        uint64_t v = big[i];
        uint32_t s = mic_isqrt(v);
        CHECK((unsigned __int128)s * s <= v && (unsigned __int128)(s + 1ull) * (s + 1ull) > v, "isqrt big %llu", (unsigned long long)v);
    }

    /* random windows against a double-precision reference */
    srand(12345);
    for (int trial = 0; trial < 200; trial++) {
        int n = 1 + rand() % 8000;
        int span = 1 + rand() % 16383;
        int base = rand() % (16384 - span);
        double sum = 0, sumsq = 0; uint32_t mn = 0xFFFFFFFFu, mx = 0;
        for (int i = 0; i < n; i++) {
            buf[i] = (uint32_t)(base + rand() % (span + 1));
            sum += buf[i]; sumsq += (double)buf[i] * buf[i];
            if (buf[i] < mn) mn = buf[i];
            if (buf[i] > mx) mx = buf[i];
        }
        double mean = sum / n;
        double sd = sqrt(sumsq / n - mean * mean);
        summarise(buf, (uint32_t)n, &r);
        CHECK(r.min == mn && r.max == mx, "random min/max trial %d", trial);
        CHECK(r.mean == (uint32_t)floor(mean), "random mean trial %d: %u vs %.3f", trial, r.mean, mean);
        CHECK(fabs((double)r.rms - sd) <= 1.0, "random rms trial %d: %u vs %.3f", trial, r.rms, sd);
    }

    /* reset really clears the window */
    mic_win_reset(&w); mic_win_add(&w, 100); mic_win_reset(&w); mic_win_add(&w, 7); mic_win_summary(&w, &r);
    CHECK(r.n == 1 && r.min == 7 && r.max == 7 && r.mean == 7, "reset");

    if (failures) { printf("%d check(s) FAILED\n", failures); return 1; }
    printf("mic_stats host test: all checks passed\n");
    return 0;
}
