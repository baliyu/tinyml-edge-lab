# CWRU bearing experiments: design notes

Everything here runs on a PC. None of it has been run on a microcontroller.

## Data
* 40 official recordings from the Case Western Reserve University Bearing Data Center, all at **48 kHz**: the four normal
  baseline files, and drive-end inner-race, ball and outer-race (centred, 6 o'clock) faults at 0.007, 0.014 and 0.021 inch,
  each at motor loads of 0 to 3 hp.
* Why 48 kHz for everything: the normal baseline files exist only at 48 kHz. Mixing them with 12 kHz fault files would let a
  model learn how a file was sampled instead of whether it has a fault.
* The manifest in `src/tinyml_lab/cwru.py` (file numbers, byte sizes, SHA-256) was generated from the `bearing-datasets`
  package's copy of the four official index pages and the checksums it recorded for the files on the official site.
  `scripts/download_cwru.py` refuses to keep a file that does not match. If a check fails, the file is corrupted or the
  official file has changed: do not use it until you know which.
* The official table skips a number: the 3 hp inner-race 0.021 inch recording is file 217, not 216.
* Licence: not stated by the data centre. Cite CWRU and Smith and Randall (2015), "Rolling element bearing diagnostics using
  the Case Western Reserve University data: A benchmark study", Mechanical Systems and Signal Processing 64-65, 100-131.

## Evaluation rules
1. A random split of windows is **leaky**: windows cut from one recording are neighbours. It is reported only to expose
   the problem.
2. Leave one motor load out is the main test. It is still easier than meeting a new bearing, because every damaged bearing
   was recorded at all four loads.
3. Leave one fault size out (three fault types, no normal class) is the harder test, because each size is a different
   damaged bearing.
4. The 400 Hz row reduces the same recordings with an anti-alias filter. It shows what a 400 Hz sensor could see of
   these signals. It says nothing about any real sensor or machine.
5. Model settings (a depth-8 tree, a 100-tree forest) were fixed in advance and not tuned.

## Not claimed
* No claim about field performance: CWRU is a small laboratory benchmark with seeded faults.
* No on-device result. The 400 Hz row is a PC experiment.
