#!/usr/bin/env bash
# edge_demo_build.sh  -  build the gesture demo for the SparkFun Edge and stage it for flashing.
#
# The demo reads the accelerometer, runs the whole gesture chain on the chip (firmware/sparkfun_edge/gesture_demo) and prints one
# line per window. It needs the SDK from edge_build.sh (in ~/AmbiqSuiteSDK) and the generated model (gesture_model.c in the
# gesture_demo folder; python scripts/export_gesture_model.py writes it). Nothing is flashed. The Ambiq/SparkFun Makefile is NOT
# copied into this repo (it has its own licence): it is derived from your SDK copy at build time.
#
#   bash firmware/sparkfun_edge/scripts/edge_demo_build.sh
set -euo pipefail
AMB_ROOT="${AMB_ROOT:-$HOME/AmbiqSuiteSDK}"
FLASH_DIR="${FLASH_DIR:-/mnt/c/edge_flash}"
BUILD_DIR="${BUILD_DIR:-$HOME/edge_demo_build}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$HERE/../gesture_demo"
FEAT="$HERE/../gesture_features"
SDK_MAKEFILE="$AMB_ROOT/boards_sfe/edge/examples/lis2dh12_accelerometer_uart/gcc/Makefile"
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

for t in make arm-none-eabi-gcc arm-none-eabi-size sed; do command -v "$t" >/dev/null || fail "$t is missing (run edge_build.sh first)."; done
for f in "$DEMO/main.c" "$DEMO/gesture_pipeline.c" "$DEMO/gesture_pipeline.h" "$FEAT/edge_features.c" "$FEAT/edge_features.h"; do
  [ -f "$f" ] || fail "cannot find $f"
done
for f in "$DEMO/gesture_model.c" "$DEMO/gesture_model.h"; do
  [ -f "$f" ] || fail "$f is missing. Write the model first:  python scripts/export_gesture_model.py"
done
[ -f "$SDK_MAKEFILE" ] || fail "the SparkFun SDK is not at $AMB_ROOT. Run firmware/sparkfun_edge/scripts/edge_build.sh first."
[ -d "$FLASH_DIR" ] || fail "$FLASH_DIR does not exist. Run edge_build.sh first."

echo "== Preparing the build in $BUILD_DIR"
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR/gcc" "$BUILD_DIR/src"
cp "$DEMO/main.c" "$DEMO/gesture_pipeline.c" "$DEMO/gesture_pipeline.h" "$DEMO/gesture_model.c" "$DEMO/gesture_model.h" "$BUILD_DIR/src/"
cp "$FEAT/edge_features.c" "$FEAT/edge_features.h" "$BUILD_DIR/src/"
# derive the Makefile: new target name, do not pick up SparkFun's own main.c, add the chain's source files, and optimise for size
# (SparkFun's examples build with -O0, which would make the model several times bigger and slower)
sed -e 's/^TARGET := lis2dh12_accelerometer_uart[[:space:]]*$/TARGET := gesture_demo/' \
    -e '/^VPATH+= \$(COMMONPATH)\/examples\/lis2dh12_accelerometer_uart[[:space:]]*$/d' \
    -e '0,/^SRC+= main\.c[[:space:]]*$/s//SRC+= main.c\nSRC+= gesture_pipeline.c\nSRC+= edge_features.c\nSRC+= gesture_model.c/' \
    -e 's/^CFLAGS+= -O0[[:space:]]*$/CFLAGS+= -Os -fno-math-errno/' \
    "$SDK_MAKEFILE" > "$BUILD_DIR/gcc/Makefile"
grep -q '^TARGET := gesture_demo' "$BUILD_DIR/gcc/Makefile"        || fail "could not rename the target: SparkFun's Makefile layout changed."
grep -q '^SRC+= gesture_model\.c' "$BUILD_DIR/gcc/Makefile"        || fail "could not add the chain's source files: SparkFun's Makefile layout changed."
grep -q '^CFLAGS+= -Os -fno-math-errno' "$BUILD_DIR/gcc/Makefile" || fail "could not set the optimisation level: SparkFun's Makefile layout changed."
! grep -q 'examples/lis2dh12_accelerometer_uart' "$BUILD_DIR/gcc/Makefile" || fail "SparkFun's own example path is still in the Makefile."

echo "== Building"
make -C "$BUILD_DIR/gcc" SDKPATH="$AMB_ROOT" COMMONPATH="$AMB_ROOT/boards_sfe/common" BOARDPATH="$AMB_ROOT/boards_sfe/edge" \
     PROJECTPATH="$BUILD_DIR" 2>&1 | grep -v "warning: you have not defined" | tail -12
AXF="$BUILD_DIR/gcc/bin/gesture_demo_asb.axf"
BIN="$BUILD_DIR/gcc/bin/gesture_demo_asb.bin"
[ -f "$BIN" ] || fail "the build did not produce $BIN. Copy the output above and send it to Claude."
cp "$BIN" "$FLASH_DIR/gesture_asb.bin"
echo
arm-none-eabi-size "$AXF"
echo "built: gesture_asb.bin ($(stat -c %s "$BIN") bytes), staged in $FLASH_DIR"

cat << MSG

NOT flashed yet. In Windows PowerShell (close any serial terminal first), same method as for the logger:

    cd C:\\edge_flash
    \$env:ASB_WAIT_SECONDS = "40"
    py asb\\asb_manual.py --bin gesture_asb.bin --load-address-blob 0x20000 --magic-num 0xCB -o gesture --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 115200 -port COM4 -r 2 -v

Press Enter first, then hold button 14, tap RST, keep holding 14 until "Upload complete!". Release 14, tap RST once more.
(The image is larger than the logger's. That is fine with this command: Ambiq's documentation allows the temporary load address, 0x20000,
to overlap the install address, 0xC000, as long as it is not below it.)
Then open the serial port at 115200 baud: it prints one "G,..." line every 1.28 s.
MSG
