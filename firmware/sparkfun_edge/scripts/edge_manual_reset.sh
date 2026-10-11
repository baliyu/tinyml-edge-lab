#!/usr/bin/env bash
# edge_manual_reset.sh  -  SparkFun Edge, step 1b: make the uploader work with a USB adapter that has NO DTR pin.
#
# Why: SparkFun's uploader resets the board itself through the adapter's DTR line, then says "Hello" within the
# bootloader's 250 ms listening window. Your CP2102 module has no DTR pin, so you reset by hand, far too slowly.
# The patched copy (asb_manual.py) keeps saying Hello (alternating 115200 and 921600 baud) until the bootloader
# answers, so your manual RST tap can land anywhere inside the waiting time. SparkFun's own files are not modified.
#
# Run it from WSL (after edge_build.sh has run):
#   bash "$(ls -t /mnt/c/Users/*/Downloads/edge_manual_reset.sh | head -1)"
set -euo pipefail
AMB_ROOT="${AMB_ROOT:-$HOME/AmbiqSuiteSDK}"
FLASH_DIR="${FLASH_DIR:-/mnt/c/edge_flash}"
SRC="$AMB_ROOT/boards_sfe/common/tools_sfe/asb/asb.py"
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }
[ -f "$SRC" ] || fail "SparkFun's uploader is not at $SRC. Run edge_build.sh first."
[ -d "$FLASH_DIR/asb" ] || fail "$FLASH_DIR/asb does not exist. Run edge_build.sh first."
command -v python3 >/dev/null || fail "python3 is not installed in WSL."

PY="$(mktemp)"; trap 'rm -f "$PY"' EXIT
cat > "$PY" << 'EOF_PATCH'
"""Make asb_manual.py from SparkFun's asb.py: keep saying Hello until the bootloader answers (for adapters with no DTR pin)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
old = "    verboseprint('Sending Hello.')\n    response = send_command(hello, 88, ser, verboseprint)\n"
if s.count(old) != 1:
    sys.exit("STOP: SparkFun's asb.py is not the version this patch was written for (anchor not found exactly once).")
new = '''    verboseprint('Sending Hello.')
    # ---- patched for USB adapters without a DTR pin (manual reset) ----
    # The bootloader only listens for about 250 ms after a reset, so instead of saying Hello once we keep saying it
    # (alternating between the two documented baud rates) until the bootloader answers.
    import os
    _wait = float(os.environ.get("ASB_WAIT_SECONDS", "0"))
    if _wait > 0:
        _old_timeout = ser.timeout
        ser.timeout = 0.05
        _bauds = [args.baud] + [b for b in (115200, 921600) if b != args.baud]
        _end = time.time() + _wait
        print("Saying Hello for up to %d seconds. NOW: hold button 14, tap RST, keep holding 14." % _wait, flush=True)
        response = False
        _n = 0
        while response == False and time.time() < _end:
            ser.baudrate = _bauds[_n % len(_bauds)]
            ser.reset_input_buffer()
            response = send_command(hello, 88, ser, lambda *a, **k: None)
            _n += 1
        ser.timeout = _old_timeout
        if response == False:
            print("The bootloader never answered. Was button 14 held while you tapped RST?")
            sys.exit(1)
        print("Bootloader answered at %d baud" % ser.baudrate, flush=True)
    else:
        response = send_command(hello, 88, ser, verboseprint)
    # ---- end of patch ----
'''
open(dst, "w", encoding="utf-8").write(s.replace(old, new))
print("wrote", dst)
EOF_PATCH
python3 "$PY" "$SRC" "$FLASH_DIR/asb/asb_manual.py"
python3 -c "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())" "$FLASH_DIR/asb/asb_manual.py" || fail "the patched file is not valid Python."
ls -l "$FLASH_DIR/asb/asb_manual.py"

cat << 'MSG'

Done. In Windows PowerShell:

  0. If a serial terminal (miniterm) is still open, press Ctrl+] to close it. Only one program can use COM4.

  1. Type these lines (the last one is long: do NOT press Enter on it yet):

        cd C:\edge_flash
        $env:ASB_WAIT_SECONDS = "40"
        py asb\asb_manual.py --bin firmware_asb.bin --load-address-blob 0x20000 --magic-num 0xCB -o firmware --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 115200 -port COM4 -r 2 -v

  2. Press Enter on it FIRST. It prints "Saying Hello for up to 40 seconds. NOW: hold button 14, tap RST, keep holding 14."

  3. Then: hold button 14, tap RST, and keep holding 14 until you see "Upload complete!". Then release 14.
MSG
