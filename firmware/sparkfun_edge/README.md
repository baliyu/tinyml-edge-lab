# SparkFun Edge (Apollo3 Blue): bring-up log

**Status: hello world verified on 2026-10-11.** SparkFun's `lis2dh12_accelerometer_uart` example was built in WSL, flashed
over a CP2102 USB-to-TTL adapter on COM4, and streams the Edge's accelerometer and temperature over serial at 115200 baud.
Nothing else has been run on this board yet (no model, no wake-word).

Sources: SparkFun Edge Hookup Guide and "Using SparkFun Edge Board with Ambiq Apollo3 SDK" (learn.sparkfun.com).

## Rules for this board
* **3.3 V only.** The board has no onboard 3.3 V regulator and is damaged by 5 V. Power it from a CR2032 coin cell or
  from a 3.3 V USB-serial adapter. Never connect a 5 V pin to it.
* The 6-pin UART header follows the SparkFun serial-breakout pinout (DTR, RXI, TXO, VCC, CTS, GND from the adapter's
  point of view). The silkscreen labels GRN and BLK mark the ends: DTR is by GRN, GND by BLK.

## Hardware used
* SparkFun Edge (Apollo3 Blue, Cortex-M4F), 6-pin header soldered.
* CP2102 "USB to TTL" module with five pins: +5V, GND, RXD, TXD, 3V3. **It has no DTR pin.**

## Wiring (checked against photos before the first power-up)
Positions are counted from the GRN end of the Edge's header, as in SparkFun's guide.

| Edge header position | Edge label | Wire to the CP2102 module |
|---|---|---|
| 1 (GRN end) | DTR | not connected |
| 2 | RXI | RXD |
| 3 | TXO | TXD |
| 4 | VCC | 3V3 (never the +5V pin) |
| 5 | CTS | not connected |
| 6 (BLK end) | GND | GND |

The module's +5V pin is left unconnected (insulate it: it is live when the USB is plugged in). No coin cell while the
adapter powers the board (my own precaution; the guide says one or the other).

## Software
* SparkFun's fork of the Ambiq Suite SDK (`sparkfun/AmbiqSuiteSDK`, commit `e280cbde3e36`, with its board-support
  submodule). About 1 GB; it is not committed here.
* `arm-none-eabi-gcc` 13.2.1 (Ubuntu package 15:13.2.rel1-2) with newlib, in WSL.
* Windows Python 3.13.6 with pyserial 3.5 and pycryptodome 3.24.0, for the upload.

## How to reproduce
From the repo root, in WSL:
```
bash firmware/sparkfun_edge/scripts/edge_build.sh          # clone the SDK, build the example, stage files in C:\edge_flash
bash firmware/sparkfun_edge/scripts/edge_manual_reset.sh   # make asb_manual.py (see "What I learned", point 1)
```
Then in Windows PowerShell (close any serial terminal first: only one program can use COM4):
```
cd C:\edge_flash
$env:ASB_WAIT_SECONDS = "40"
py asb\asb_manual.py --bin firmware_asb.bin --load-address-blob 0x20000 --magic-num 0xCB -o firmware --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 115200 -port COM4 -r 2 -v
```
**Press Enter first.** Then hold button 14, tap RST, and keep holding 14 until it prints `Upload complete!`.
To read the output: `py -m serial.tools.miniterm COM4 115200`, then tap RST (without 14). Ctrl+] quits.

## What I learned
1. **SparkFun's uploader expects the adapter to reset the board itself.** `tools_sfe/asb/asb.py` pulses reset through
   the DTR line, waits 100 ms and sends "Hello", and a comment in its code says the bootloader only listens for about
   250 ms. A CP2102 module has no DTR pin, so resetting by hand is far too slow: two attempts (115200 and 921600 baud)
   both ended with `No response for command` and `Upload failed`. An earlier version of this README said the book's
   manual flow (hold 14, press RST, run the uploader) works; with this tool and this adapter it does not.
2. **Fix:** `edge_manual_reset.sh` makes `asb_manual.py`, a patched copy of the uploader that keeps sending Hello
   (alternating 115200 and 921600 baud) for up to `ASB_WAIT_SECONDS`. SparkFun's own files are not modified. It was tested
   against a simulated board first; on the real board it connected at 921600 baud on the first try.
3. SparkFun's tutorial is out of date: it names `ambiq_bin2board.py`, which is no longer in the SDK. The Makefile's
   `bootload_asb` target now calls `tools_sfe/asb/asb.py`.

## Evidence
`evidence/accelerometer_uart_excerpts.txt` holds excerpts of the serial output, copied as printed.
* `Whoami (should be 0x33): 0x33` (the LIS2DH12's chip ID).
* At rest the three axes combine to about 1.05 g (for example 162, -429, 943 mg), i.e. gravity within about 5%.
  The sensor is not calibrated here.
* Tilting the board moved x from about +160 to -316 mg and y from about -430 to +256 mg.
* The temperature field read 29 to 32 degrees C (the sensor's own coarse internal reading).

## Not done yet
* The sample rate of this example was not measured.
* The factory speech demo was overwritten and has not been restored (SparkFun's `tensorflow_micro_speech` example can
  restore it with the same flashing method).
* The patched uploader has worked once, with one board and one adapter.
* No machine learning has run on this board yet.
