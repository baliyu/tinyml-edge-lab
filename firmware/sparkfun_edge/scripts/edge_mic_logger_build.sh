#!/usr/bin/env bash
# edge_mic_logger_build.sh  -  build the microphone feature logger for the SparkFun Edge and stage it for flashing.
# Built with AI assistance (Claude); derived from edge_mic_build.sh.
#
# Needs the SDK from edge_build.sh (in ~/AmbiqSuiteSDK). Nothing is flashed. The Ambiq/SparkFun Makefile is NOT copied
# into this repo (it has its own licence): it is derived from your SDK copy at build time, exactly as edge_mic_build.sh does.
#
#   bash firmware/sparkfun_edge/scripts/edge_mic_logger_build.sh
set -euo pipefail
AMB_ROOT="${AMB_ROOT:-$HOME/AmbiqSuiteSDK}"
FLASH_DIR="${FLASH_DIR:-/mnt/c/edge_flash}"
BUILD_DIR="${BUILD_DIR:-$HOME/edge_mic_logger_build}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/../mic_logger"
FEAT_DIR="$HERE/../mic_features"
SDK_MAKEFILE="$AMB_ROOT/boards_sfe/edge/examples/lis2dh12_accelerometer_uart/gcc/Makefile"
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

for t in make arm-none-eabi-gcc sed; do command -v "$t" >/dev/null || fail "$t is missing (run edge_build.sh first)."; done
[ -f "$SRC_DIR/main.c" ] || fail "cannot find $SRC_DIR/main.c"
for f in mic_features.c mic_features.h; do [ -f "$FEAT_DIR/$f" ] || fail "cannot find $FEAT_DIR/$f"; done
[ -f "$SDK_MAKEFILE" ] || fail "the SparkFun SDK is not at $AMB_ROOT. Run firmware/sparkfun_edge/scripts/edge_build.sh first."
[ -d "$FLASH_DIR" ] || fail "$FLASH_DIR does not exist. Run edge_build.sh first."

echo "== Preparing the build in $BUILD_DIR"
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR/gcc" "$BUILD_DIR/src"
cp "$SRC_DIR/main.c" "$FEAT_DIR/mic_features.c" "$FEAT_DIR/mic_features.h" "$BUILD_DIR/src/"
# derive the Makefile: new target name, and do not pick up SparkFun's own main.c
sed -e 's/^TARGET := lis2dh12_accelerometer_uart[[:space:]]*$/TARGET := mic_logger/' \
    -e '/^VPATH+= \$(COMMONPATH)\/examples\/lis2dh12_accelerometer_uart[[:space:]]*$/d' \
    "$SDK_MAKEFILE" > "$BUILD_DIR/gcc/Makefile"
grep -q '^TARGET := mic_logger' "$BUILD_DIR/gcc/Makefile"                    || fail "could not rename the target: SparkFun's Makefile layout changed."
! grep -q 'examples/lis2dh12_accelerometer_uart' "$BUILD_DIR/gcc/Makefile"   || fail "SparkFun's own example path is still in the Makefile."

echo "== Building (the first build of new code may show compiler errors: copy ALL of them and send them to Claude)"
make -C "$BUILD_DIR/gcc" SDKPATH="$AMB_ROOT" COMMONPATH="$AMB_ROOT/boards_sfe/common" BOARDPATH="$AMB_ROOT/boards_sfe/edge" \
     PROJECTPATH="$BUILD_DIR" 2>&1 | grep -v "warning: you have not defined" | tail -40
BIN="$BUILD_DIR/gcc/bin/mic_logger_asb.bin"
[ -f "$BIN" ] || fail "the build did not produce $BIN. Copy the output above and send it to Claude."
cp "$BIN" "$FLASH_DIR/miclog_asb.bin"
echo "built: miclog_asb.bin ($(stat -c %s "$BIN") bytes), staged in $FLASH_DIR"

cat << MSG

NOT flashed yet. In Windows PowerShell (close any serial terminal first), same method as the logger:

    cd C:\\edge_flash
    \$env:ASB_WAIT_SECONDS = "40"
    py asb\\asb_manual.py --bin miclog_asb.bin --load-address-blob 0x20000 --magic-num 0xCB -o miclog --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 115200 -port COM4 -r 2 -v

Press Enter first, then hold button 14, tap RST, keep holding 14 until "Upload complete!". Release 14, tap RST once more.
Then open COM4 at 115200 baud in PuTTY: you should see one line every 0.25 s.
MSG
