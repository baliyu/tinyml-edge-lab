/* Window features for the desk-gesture classifier, in plain C99: no malloc, no libm beyond sqrt.
 *
 * Port of tinyml_lab.features.window_features / tinyml_lab.edge_gestures.feature_matrix. The Python version is the
 * reference: tests/test_edge_features_c.py compiles this file and compares it with Python on real and synthetic windows.
 *
 * Build as float (what the board runs, the default) or as double (used on the PC to prove the algorithm matches Python):
 *     gcc -DEF_REAL=double -c edge_features.c
 *
 * Input: one window of EF_WINDOW samples, x/y/z interleaved (x0,y0,z0,x1,y1,z1,...), in milli-g, sampled at EF_FS_HZ.
 * Output: the features in exactly the order of tinyml_lab.edge_gestures.feature_names_for("raw") / ("motion").
 */
#ifndef EDGE_FEATURES_H
#define EDGE_FEATURES_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EF_REAL
#define EF_REAL float
#endif
typedef EF_REAL ef_real;

#define EF_WINDOW 128               /* samples per window (2.56 s at 50 Hz); must be a power of two <= the cosine table */
#define EF_FS_HZ 50.0               /* the sampling rate the features assume */
#define EF_BANDS 4                  /* spectral bands, equal width from 0 to fs/2 */
#define EF_CHANNEL_FEATURES 10      /* per channel: mean, std, rms, min, max, dom_freq, band0..band3 */
#define EF_RAW_FEATURES 40          /* x, y, z and the vector magnitude, as measured */
#define EF_MOTION_FEATURES 37       /* the window mean removed first; x, y, z means are not used */

void ef_raw_features(const ef_real *xyz, ef_real *out);       /* out: EF_RAW_FEATURES values */
void ef_motion_features(const ef_real *xyz, ef_real *out);    /* out: EF_MOTION_FEATURES values */

#ifdef __cplusplus
}
#endif
#endif
