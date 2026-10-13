/* Window features for the desk-gesture classifier. See edge_features.h. */
#include "edge_features.h"

#include <math.h>

#define EF_BINS (EF_WINDOW / 2 + 1)                    /* rfft bins: 0 .. N/2 */
#define EF_MASK (EF_WINDOW - 1)
#define EF_SIN_SHIFT (EF_WINDOW - EF_WINDOW / 4)       /* sin(2*pi*i/N) = cos(2*pi*(i + 3N/4)/N) */
#define EF_SQRT(x) _Generic((x), float: sqrtf, default: sqrt)(x)

/* A channel whose spectral energy is below this fraction of its total energy has no spectrum worth reading: what is left
 * is rounding dust (a perfectly constant magnitude, a stuck sensor). Without this, the dominant frequency and the band
 * fractions of such a channel would be decided by the order in which the sums happen to be rounded. Real sensor noise
 * (about 1 milli-g) is many orders of magnitude above it, so real windows are not affected. */
#define EF_DUST 1e-9

/* cos(2*pi*i/128), i = 0..127. A test regenerates this table with numpy and compares it with the text below. */
static const ef_real COS_TAB[EF_WINDOW] = {
    1, 0.99879545620517241, 0.99518472667219693, 0.98917650996478101,
    0.98078528040323043, 0.97003125319454397, 0.95694033573220882, 0.94154406518302081,
    0.92387953251128674, 0.90398929312344334, 0.88192126434835505, 0.85772861000027212,
    0.83146961230254524, 0.80320753148064494, 0.77301045336273699, 0.74095112535495911,
    0.70710678118654757, 0.67155895484701833, 0.63439328416364549, 0.59569930449243347,
    0.55557023301960229, 0.51410274419322166, 0.47139673682599781, 0.4275550934302822,
    0.38268343236508984, 0.33688985339222005, 0.29028467725446233, 0.24298017990326398,
    0.19509032201612833, 0.14673047445536175, 0.09801714032956077, 0.049067674327418126,
    6.123233995736766e-17, -0.049067674327418008, -0.098017140329560645, -0.14673047445536164,
    -0.19509032201612819, -0.24298017990326387, -0.29028467725446216, -0.33688985339221994,
    -0.38268343236508973, -0.42755509343028186, -0.4713967368259977, -0.51410274419322166,
    -0.55557023301960196, -0.59569930449243336, -0.63439328416364538, -0.67155895484701844,
    -0.70710678118654746, -0.74095112535495888, -0.77301045336273699, -0.80320753148064483,
    -0.83146961230254535, -0.85772861000027201, -0.88192126434835494, -0.90398929312344334,
    -0.92387953251128674, -0.9415440651830207, -0.95694033573220882, -0.97003125319454397,
    -0.98078528040323043, -0.98917650996478101, -0.99518472667219682, -0.99879545620517241,
    -1, -0.99879545620517241, -0.99518472667219693, -0.98917650996478101,
    -0.98078528040323043, -0.97003125319454397, -0.95694033573220894, -0.94154406518302081,
    -0.92387953251128685, -0.90398929312344345, -0.88192126434835505, -0.85772861000027212,
    -0.83146961230254546, -0.80320753148064494, -0.7730104533627371, -0.74095112535495911,
    -0.70710678118654768, -0.67155895484701866, -0.63439328416364593, -0.59569930449243313,
    -0.55557023301960218, -0.51410274419322177, -0.47139673682599786, -0.42755509343028247,
    -0.38268343236509034, -0.33688985339221994, -0.29028467725446244, -0.24298017990326412,
    -0.19509032201612866, -0.1467304744553623, -0.098017140329560451, -0.049067674327418029,
    -1.8369701987210297e-16, 0.049067674327417661, 0.09801714032956009, 0.14673047445536194,
    0.1950903220161283, 0.24298017990326376, 0.29028467725446205, 0.33688985339221961,
    0.38268343236509, 0.42755509343028214, 0.47139673682599759, 0.51410274419322155,
    0.55557023301960184, 0.59569930449243291, 0.6343932841636456, 0.67155895484701833,
    0.70710678118654735, 0.74095112535495888, 0.77301045336273666, 0.80320753148064505,
    0.83146961230254524, 0.85772861000027201, 0.88192126434835483, 0.90398929312344312,
    0.92387953251128652, 0.94154406518302081, 0.95694033573220882, 0.97003125319454397,
    0.98078528040323032, 0.9891765099647809, 0.99518472667219693, 0.99879545620517241,
};

/* The spectral band edges, in rfft bins, for EF_BANDS equal-width bands over the EF_BINS bins: the same integers as
 * numpy.linspace(0, EF_BINS, EF_BANDS + 1).astype(int) = 0, 16, 32, 48, 65 for 128 samples and 4 bands. */
static int band_edge(int b)
{
    return (int)(((long)b * EF_BINS) / EF_BANDS);
}

/* Features of one channel, in the order mean, std, rms, min, max, dom_freq, band0..band3, exactly as
 * tinyml_lab.features.window_features computes them. */
static void channel_features(const ef_real *x, ef_real *f)
{
    ef_real xc[EF_WINDOW];
    ef_real spec[EF_BINS];
    ef_real sum = 0, sumsq = 0, mn = x[0], mx = x[0], var = 0, total = 0, best = 0;
    int i, k, b, dom_bin = 0;

    for (i = 0; i < EF_WINDOW; i++) {
        sum += x[i];
        sumsq += x[i] * x[i];
        if (x[i] < mn) mn = x[i];
        if (x[i] > mx) mx = x[i];
    }
    const ef_real mean = sum / EF_WINDOW;
    for (i = 0; i < EF_WINDOW; i++) {
        xc[i] = x[i] - mean;
        var += xc[i] * xc[i];
    }

    /* Power spectrum of the mean-removed window by a direct transform, DC dropped. 64 bins x 128 samples = 8192 steps,
     * a few milliseconds on the Edge, once per window. */
    spec[0] = 0;
    for (k = 1; k < EF_BINS; k++) {
        ef_real re = 0, im = 0;
        unsigned idx = 0;                              /* (k * i) mod N, kept without a multiply */
        for (i = 0; i < EF_WINDOW; i++) {
            re += xc[i] * COS_TAB[idx];
            im += xc[i] * COS_TAB[(idx + EF_SIN_SHIFT) & EF_MASK];
            idx = (idx + (unsigned)k) & EF_MASK;
        }
        spec[k] = re * re + im * im;
        total += spec[k];
        if (spec[k] > best) {                          /* strict: the first of equal maxima wins, like numpy.argmax; in
                                                          practice a tie is decided by rounding, in Python as much as here */
            best = spec[k];
            dom_bin = k;
        }
    }

    const int has_spectrum = total > (ef_real)EF_DUST * sumsq;

    f[0] = mean;
    f[1] = EF_SQRT(var / EF_WINDOW);
    f[2] = EF_SQRT(sumsq / EF_WINDOW);
    f[3] = mn;
    f[4] = mx;
    f[5] = has_spectrum ? (ef_real)dom_bin * (ef_real)(EF_FS_HZ / EF_WINDOW) : (ef_real)0;
    for (b = 0; b < EF_BANDS; b++) {
        ef_real s = 0;
        for (k = band_edge(b); k < band_edge(b + 1); k++) s += spec[k];
        f[6 + b] = has_spectrum ? s / total : (ef_real)0;
    }
}

static void fill_channels(const ef_real *xyz, ef_real ch[4][EF_WINDOW], int remove_mean)
{
    ef_real m[3] = {0, 0, 0};
    int i, a;

    if (remove_mean) {
        for (i = 0; i < EF_WINDOW; i++)
            for (a = 0; a < 3; a++) m[a] += xyz[3 * i + a];
        for (a = 0; a < 3; a++) m[a] /= EF_WINDOW;
    }
    for (i = 0; i < EF_WINDOW; i++) {
        for (a = 0; a < 3; a++) ch[a][i] = xyz[3 * i + a] - m[a];
        const ef_real mag2 = ch[0][i] * ch[0][i] + ch[1][i] * ch[1][i] + ch[2][i] * ch[2][i];
        ch[3][i] = EF_SQRT(mag2);
    }
}

void ef_raw_features(const ef_real *xyz, ef_real *out)
{
    ef_real ch[4][EF_WINDOW];
    int c;

    fill_channels(xyz, ch, 0);
    for (c = 0; c < 4; c++) channel_features(ch[c], out + c * EF_CHANNEL_FEATURES);
}

void ef_motion_features(const ef_real *xyz, ef_real *out)
{
    ef_real ch[4][EF_WINDOW];
    ef_real f[EF_CHANNEL_FEATURES];
    int c, j, o = 0;

    fill_channels(xyz, ch, 1);
    for (c = 0; c < 3; c++) {                          /* x, y, z: the mean is zero by construction and is not output */
        channel_features(ch[c], f);
        for (j = 1; j < EF_CHANNEL_FEATURES; j++) out[o++] = f[j];
    }
    channel_features(ch[3], f);                        /* the magnitude of what is left: all ten, its mean is not zero */
    for (j = 0; j < EF_CHANNEL_FEATURES; j++) out[o++] = f[j];
}
