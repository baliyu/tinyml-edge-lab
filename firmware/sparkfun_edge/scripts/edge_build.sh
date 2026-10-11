#!/usr/bin/env bash
# edge_build.sh  -  SparkFun Edge, step 1: build SparkFun's accelerometer example in WSL and stage it for flashing from Windows.
#
# What it does (nothing is flashed, no board needed):
#   1. checks your tools (git, make, arm-none-eabi-gcc, the C library) and free disk space
#   2. gets SparkFun's Ambiq SDK (about 1 GB, pinned to the commit I tested) into ~/AmbiqSuiteSDK
#   3. builds the example "lis2dh12_accelerometer_uart" (reads the Edge's accelerometer and prints it over serial)
#   4. copies the firmware image and the small upload tool to C:\edge_flash so Windows Python can flash it over COM4
#
# Run it from WSL:
#   bash "$(ls -t /mnt/c/Users/*/Downloads/edge_build.sh | head -1)"
set -euo pipefail

AMB_ROOT="${AMB_ROOT:-$HOME/AmbiqSuiteSDK}"
FLASH_DIR="${FLASH_DIR:-/mnt/c/edge_flash}"
EXAMPLE="${EXAMPLE:-lis2dh12_accelerometer_uart}"
SDK_COMMIT="e280cbde3e366509da6768ab95471782a05d2371"
say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

say "1/5 Checking your tools"
for t in git make arm-none-eabi-gcc arm-none-eabi-objcopy; do
  command -v "$t" >/dev/null || fail "$t is missing. Install it with:  sudo apt update && sudo apt install -y git make gcc-arm-none-eabi libnewlib-arm-none-eabi"
done
echo "compiler: $(arm-none-eabi-gcc --version | head -1)"
libc="$(arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -print-file-name=libc.a)"
[ "$libc" != "libc.a" ] || fail "the C library for the compiler is missing. Run:  sudo apt install -y libnewlib-arm-none-eabi"
if [ ! -d "$AMB_ROOT/.git" ] && [ ! -d "$AMB_ROOT/boards_sfe" ]; then
  free_mb="$(df -Pm "$HOME" | awk 'NR==2 {print $4}')"
  [ "$free_mb" -ge 3000 ] || fail "only ${free_mb} MB free in your WSL home; the SDK needs about 3000 MB while it downloads."
fi

say "2/5 Getting SparkFun's Ambiq SDK (about 1 GB, this takes a few minutes the first time)"
if [ -d "$AMB_ROOT/boards_sfe/edge/examples/$EXAMPLE" ]; then
  echo "already there: $AMB_ROOT"
else
  git clone --recurse-submodules --shallow-submodules https://github.com/sparkfun/AmbiqSuiteSDK "$AMB_ROOT"
  got="$(git -C "$AMB_ROOT" rev-parse HEAD)"
  if [ "$got" != "$SDK_COMMIT" ]; then
    echo "note: the SDK is at $got, not the commit I tested ($SDK_COMMIT). Continuing; tell Claude if the build fails."
  fi
fi

say "3/5 Building the example ($EXAMPLE)"
cd "$AMB_ROOT/boards_sfe/edge/examples/$EXAMPLE/gcc"
make clean >/dev/null 2>&1 || true
make 2>&1 | grep -v "warning: you have not defined" | tail -12
BIN="bin/${EXAMPLE}_asb.bin"
[ -f "$BIN" ] || fail "the build did not produce $BIN. Copy the output above and send it to Claude."
echo "built: $BIN ($(stat -c %s "$BIN") bytes)"

say "4/5 Staging for Windows in $FLASH_DIR"
mkdir -p "$FLASH_DIR/asb"
cp "$BIN" "$FLASH_DIR/firmware_asb.bin"
cp "$AMB_ROOT/boards_sfe/common/tools_sfe/asb/"{asb.py,am_defines.py,keys_info.py} "$FLASH_DIR/asb/"
ls -l "$FLASH_DIR" "$FLASH_DIR/asb" | sed 's/^/   /'

say "5/5 Done. Next steps are in Windows (PowerShell), not WSL"
cat <<'MSG'

The firmware is built and staged. It is NOT on the board yet.

In Windows PowerShell:

  1. One-time: check that Windows has Python, and install the two packages the uploader needs:
        py --version
        py -m pip install pyserial pycryptodome
     (If "py" is not found, tell Claude and we will install Python first.)

  2. Make sure no serial terminal is open on COM4. Only one program can use the port at a time.

  3. Tell Claude when you are ready to flash. The flash needs both hands:
     hold button 14, tap RST, press Enter on the command, release 14 when the upload finishes.
MSG
