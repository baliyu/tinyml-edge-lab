"""Read the memory figures from a saved arduino-cli compile summary.

AI-assisted: drafted with Claude, checked by the author.
"""
import re


def parse_compile_log(text):
    flash = re.search(
        r"Sketch uses (\d+) bytes \((\d+)%\) of program storage space\. Maximum is (\d+) bytes",
        text,
    )
    ram = re.search(r"Global variables use (\d+) bytes", text)
    if flash is None or ram is None:
        raise ValueError("compile log has no memory summary")
    return {
        "flash_bytes": int(flash.group(1)),
        "flash_max": int(flash.group(3)),
        "static_ram_bytes": int(ram.group(1)),
    }
