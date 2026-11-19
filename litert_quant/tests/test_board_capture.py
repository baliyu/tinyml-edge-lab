import itertools

from litert_quant.board_capture import read_run


class FakeSerial:
    def __init__(self, lines):
        self.queue = [line.encode() + b"\n" for line in lines]
        self.written = b""

    def readline(self):
        return self.queue.pop(0) if self.queue else b""

    def write(self, data):
        self.written += data


def test_read_run_stops_at_done():
    ser = FakeSerial(["START,int8,1", "S,0,1,1,100", "DONE,int8,3000,1,1,5640", "extra"])
    lines = read_run(ser, timeout_s=10)
    assert lines[-1].startswith("DONE,")
    assert len(lines) == 3
    assert len(ser.queue) == 1


def test_read_run_sends_r_when_asked():
    ser = FakeSerial(["DONE,int8,1,1,1,1"])
    read_run(ser, timeout_s=10, send_r=True)
    assert ser.written == b"r"


def test_read_run_times_out():
    ser = FakeSerial([])
    ticks = itertools.count(0.0, 1.0)
    lines = read_run(ser, timeout_s=3, clock=lambda: next(ticks))
    assert lines == []
