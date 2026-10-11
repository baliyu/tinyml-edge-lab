#!/usr/bin/env bash
# edge_logger_build.sh  -  build the fixed-rate CSV accelerometer logger for the SparkFun Edge and stage it for flashing.
#
# Needs the SDK from edge_build.sh (in ~/AmbiqSuiteSDK). Nothing is flashed. The Ambiq/SparkFun Makefile is NOT copied
# into this repo (it has its own licence): it is derived from your SDK copy at build time.
#
#   bash firmware/sparkfun_edge/scripts/edge_logger_build.sh            # 50 Hz (default)
#   ODR_HZ=100 bash firmware/sparkfun_edge/scripts/edge_logger_build.sh # 25, 50 or 100
set -euo pipefail
AMB_ROOT="${AMB_ROOT:-$HOME/AmbiqSuiteSDK}"
FLASH_DIR="${FLASH_DIR:-/mnt/c/edge_flash}"
BUILD_DIR="${BUILD_DIR:-$HOME/edge_logger_build}"
ODR_HZ="${ODR_HZ:-50}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_MAIN="$HERE/../imu_logger/main.c"
SDK_MAKEFILE="$AMB_ROOT/boards_sfe/edge/examples/lis2dh12_accelerometer_uart/gcc/Makefile"
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

case "$ODR_HZ" in 25|50|100) ;; *) fail "ODR_HZ must be 25, 50 or 100 (got '$ODR_HZ').";; esac
for t in make arm-none-eabi-gcc sed; do command -v "$t" >/dev/null || fail "$t is missing (run edge_build.sh first)."; done
[ -f "$SRC_MAIN" ] || fail "cannot find $SRC_MAIN"
[ -f "$SDK_MAKEFILE" ] || fail "the SparkFun SDK is not at $AMB_ROOT. Run firmware/sparkfun_edge/scripts/edge_build.sh first."
[ -d "$FLASH_DIR" ] || fail "$FLASH_DIR does not exist. Run edge_build.sh first."

echo "== Preparing the build in $BUILD_DIR (ODR_HZ=$ODR_HZ)"
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR/gcc" "$BUILD_DIR/src"
cp "$SRC_MAIN" "$BUILD_DIR/src/main.c"
# derive the Makefile: new target name, do not pick up SparkFun's own main.c, pass the sample rate to the compiler
sed -e 's/^TARGET := lis2dh12_accelerometer_uart[[:space:]]*$/TARGET := imu_logger/' \
    -e '/^VPATH+= \$(COMMONPATH)\/examples\/lis2dh12_accelerometer_uart[[:space:]]*$/d' \
    -e '0,/^DEFINES+=[[:space:]]*$/s//DEFINES+= -DODR_HZ=$(ODR_HZ)/' \
    "$SDK_MAKEFILE" > "$BUILD_DIR/gcc/Makefile"
grep -q '^TARGET := imu_logger' "$BUILD_DIR/gcc/Makefile"       || fail "could not rename the target: SparkFun's Makefile layout changed."
grep -q 'DEFINES+= -DODR_HZ=' "$BUILD_DIR/gcc/Makefile"          || fail "could not set the sample rate: SparkFun's Makefile layout changed."
! grep -q 'examples/lis2dh12_accelerometer_uart' "$BUILD_DIR/gcc/Makefile" || fail "SparkFun's own example path is still in the Makefile."

echo "== Building"
make -C "$BUILD_DIR/gcc" SDKPATH="$AMB_ROOT" COMMONPATH="$AMB_ROOT/boards_sfe/common" BOARDPATH="$AMB_ROOT/boards_sfe/edge" \
     PROJECTPATH="$BUILD_DIR" ODR_HZ="$ODR_HZ" 2>&1 | grep -v "warning: you have not defined" | tail -12
BIN="$BUILD_DIR/gcc/bin/imu_logger_asb.bin"
[ -f "$BIN" ] || fail "the build did not produce $BIN. Copy the output above and send it to Claude."
cp "$BIN" "$FLASH_DIR/logger_asb.bin"
echo "built: logger_asb.bin ($(stat -c %s "$BIN") bytes), staged in $FLASH_DIR"

cat << MSG

NOT flashed yet. In Windows PowerShell (close any serial terminal first), same method as before:

    cd C:\\edge_flash
    \$env:ASB_WAIT_SECONDS = "40"
    py asb\\asb_manual.py --bin logger_asb.bin --load-address-blob 0x20000 --magic-num 0xCB -o logger --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 115200 -port COM4 -r 2 -v

Press Enter first, then hold button 14, tap RST, keep holding 14 until "Upload complete!". Release 14, tap RST once more.
MSG
