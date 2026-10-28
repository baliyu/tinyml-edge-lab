# mic_probe

A first look at the SparkFun Edge's analog microphone, before any machine learning. It samples microphone MIC1 (GPIO 29, ADC channel 1) at about 8000 Hz and prints one line per second over the serial port: how many samples arrived, the smallest and largest value, the average, a loudness number, and the measured sample rate.

Built with AI assistance (Claude). The program is adapted from the Ambiq SDK example `adc_lpmode0_dma` (timer-triggered ADC with DMA) and from `imu_logger`.

## What is tested

- `mic_stats.c` (the per-second statistics) has a host test: `gcc -Wall -Wextra -Werror -O2 host_test.c mic_stats.c -lm -o host_test && ./host_test`. It checks constant, alternating and full-scale signals, an empty window, an exact integer square root, and 200 random windows against a double-precision reference. Seven deliberate bugs in `mic_stats.c` were each caught by the test.
- `main.c` compiles with no errors or warnings and has been run on one SparkFun Edge, over serial:
  - Quiet: `n` was 8000 every second, `mean` about 8440, `rms` about 145 to 150.
  - With a 12 MHz timer constant, `rate_hz` read about 7447. A phone stopwatch agreed: 60 lines took 64.61 s. With `CTIMER_HZ` set to 11,170,500, `rate_hz` reads 8000 to 8001. That value is measured on this one board. The cause of the difference is not known.
  - Clipping: sharp sounds such as claps, even soft ones about 1 m away, reach 16383 with the 2.0 V reference. Speech in one test peaked at 15331 and did not clip.
- Not done: no second board, no temperature check, and no training data recorded yet.

## Build and flash

```
bash firmware/sparkfun_edge/scripts/edge_mic_build.sh
```

then follow the PowerShell lines it prints (same method as the logger). Open the Edge's serial port at 115200 baud.

## What to look for

- `n` should be 8000 every second, and `rate_hz` close to 8000. If not, samples are being lost or the timer period is wrong.
- `mean` is the microphone's resting level. `min` and `max` should sit well away from 0 and 16383; if `max` reaches 16383, the signal clips.
- `rms` should be small in a quiet room and jump when you clap or tap the board.

## Limits
One microphone (MIC1), one board, one sample rate, no gain tuning. The 2.0 V reference was a guess, and the first run showed it is too tight for sharp sounds such as claps. The timer constant was corrected after the first run, on this board only.
