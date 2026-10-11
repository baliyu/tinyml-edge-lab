# Edge IMU logger (fixed-rate CSV, 50 Hz)

**Status: flashed and streaming on the board (first run seen in a terminal on 2026-10-11, see `../evidence/`); a labelled
capture checked with `tools/edge_capture.py` is still to come.** This file will say "verified" only after that.

A small firmware for the SparkFun Edge's LIS2DH12 accelerometer that prints one CSV line per sample, plus a PC tool that
turns the stream into labelled files and reports how good each capture was. Purpose: collect my own labelled motion data
at 50 Hz, the rate of the UCI HAR data the models in `src/tinyml_lab/` were trained on. (SparkFun's stock example is
configured for 25 Hz: `LIS2DH12_ODR_25Hz` in its source.)

`main.c` is derived from SparkFun's `lis2dh12_accelerometer_uart` example (MIT licence, notice kept at the top of the file).
SparkFun's/Ambiq's Makefile has its own licence and is **not** copied here: the build script derives it from your SDK copy.

## What the firmware prints (115200 baud)
```
# sparkfun_edge lis2dh12 logger v1, whoami 0x33
# odr_hz=50 fs_g=2 mode=hr12 units=mg tick_hz=32768 timer=xtal32k
t_ticks,x_mg,y_mg,z_mg
12345678,-541,507,544
...
# stats samples=500 overruns=0 i2c_err=0 rate_hz=49.987 tick_hz=32768 timer=xtal32k
```
* Lines starting with `#` are comments. `t_ticks` is a free-running 32-bit hardware timer (it wraps; the PC tool unwraps it).
  Divide by `tick_hz` for seconds.
* Values are integer milli-g (+-2 g, 12-bit high-resolution mode, 1 mg per count).
* Every 10 s the firmware prints the sample rate **it measured from its own timestamps**, plus the sensor's overrun count
  (samples overwritten before they were read) and I2C errors.
* The board-support code stops the 32 kHz crystal, so the firmware restarts it and checks that it really counts. If it
  does not, it falls back to the internal 3 MHz oscillator (less accurate) and says so in the `timer=` field.

## Build, flash, capture
1. Build (WSL, needs the SDK from `edge_build.sh`; nothing is flashed):
   ```
   bash firmware/sparkfun_edge/scripts/edge_logger_build.sh            # 50 Hz. ODR_HZ=25 or ODR_HZ=100 also work
   ```
2. Flash from Windows PowerShell with the manual-reset uploader (see `../README.md`), using `logger_asb.bin`:
   ```
   cd C:\edge_flash
   $env:ASB_WAIT_SECONDS = "40"
   py asb\asb_manual.py --bin logger_asb.bin --load-address-blob 0x20000 --magic-num 0xCB -o logger --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 115200 -port COM4 -r 2 -v
   ```
   Press Enter first, then hold button 14, tap RST, and keep holding 14 until `Upload complete!`.
3. Capture (Windows PowerShell, from the repo root; close any serial terminal first):
   ```
   py -m pip install pyserial
   py tools\edge_capture.py --port COM4 --label rest --seconds 30
   ```
   Files go to `data/own/edge/` (ignored by git: commit only the few you want to keep as evidence).
   Do not click inside the console while it records: selecting text pauses a Windows console, the adapter's buffer
   overflows and samples are lost.

## Checking a capture
`edge_capture.py` prints and saves: the measured rate, the number of gaps in the timestamps, the firmware's overrun and
I2C error counts, and unreadable lines. The verdict is `OK` only if the rate is within 2% of the configured rate and
nothing was lost. A first real capture should be checked by hand too: lying flat and still, the three axes should combine to
about 1000 mg, and the measured rate should be close to 50 Hz.

## Limits
* The firmware polls the sensor's data-ready flag instead of using its interrupt or FIFO. At 50 Hz this has plenty of
  margin, but a stall of more than 20 ms (for example a very slow serial host) would lose samples; the overrun counter
  would show it.
* Timestamps are when the MCU read each sample, so they jitter by a millisecond or so around the sensor's own timing.
  The measured rate is an average over the capture.
* The sensor's output rate comes from an internal oscillator. I found no published accuracy for the LIS2DH12 (an ST
  community thread from 2016 reports about 2.8% spread between parts). On this unit the firmware measured
  51.22 Hz for a configured 50 Hz. So use the saved timestamps, not the sample count, and resample to exactly 50 Hz
  before comparing with the UCI data: `edge_capture.py` writes a resampled file for that.
* The measured rate is only as accurate as the timer named in `timer=`: the 32 kHz crystal (`xtal32k`) is far
  better than the sensor's own clock; the fallback (`hfrc3mhz`) is not.
* Only the accelerometer is used. The Edge's microphones and camera are not touched.

## First observations (2026-10-11, one run, not yet a checked capture)
Copied from the terminal into `../evidence/logger_first_run_excerpts.txt`:
* `timer=xtal32k`: the crystal restarted and counts, so the timestamps use the accurate clock.
* The firmware's own statistics line after 500 samples: `overruns=1 i2c_err=0 rate_hz=51.220`.
* Consecutive timestamps alternate between about 631 and 654 ticks (19.3 and 20.0 ms): the polling loop takes about
  0.7 ms per pass, so each sample is stamped up to one pass late.
* Lying flat, the three axes combine to about 1.04 g (for example 45, -60, 1042 mg).
* The single overrun is unexplained. One possibility is start-up (the three header lines take about 15 ms to print and
  the sensor is already sampling), but that is a guess. A capture of 30 s or more shows whether the count grows; the tool
  only reports overruns that happen during the capture.
