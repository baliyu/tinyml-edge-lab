"""Shared fakes for the live accuracy tests: a clock, a board that prints window lines, and synthetic runs. Nothing here is real data."""
from tinyml_lab import live_accuracy as la

STEP = 1.28
TICKS_PER_STEP = 3_840_000


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class Interrupt(Exception):
    pass


class FakeBoard:
    """A serial port on which the board prints one window line every 1.28 s of fake time.

    predict(t_rel) gives the gesture for the window whose middle is t_rel seconds after the first window arrived. stall=(a, b) makes the PC
    not read between a and b (relative seconds): lines due in between arrive together at b. interrupt_at raises KeyboardInterrupt once the clock passes it.
    """

    def __init__(self, clock, predict, first_due=0.5, stall=None, interrupt_at=None, never=False, ticks0=4_290_000_000, timeout=0.2):
        self.clock, self.predict, self.stall, self.interrupt_at, self.never = clock, predict, stall, interrupt_at, never
        self.t_start, self.first_due, self.n, self.ticks0, self.timeout = clock.t, first_due, 0, ticks0, timeout
        self.resets = 0

    def reset_input_buffer(self):
        self.resets += 1

    def readline(self):
        if self.interrupt_at is not None and self.clock.t - self.t_start >= self.interrupt_at:
            raise KeyboardInterrupt
        if self.never:
            self.clock.t += self.timeout
            return b""
        due = self.first_due + self.n * STEP                       # relative to when the board was created
        deliver = due
        if self.stall and self.stall[0] <= due <= self.stall[1]:
            deliver = self.stall[1]
        if self.t_start + deliver <= self.clock.t + self.timeout:
            self.clock.t = max(self.clock.t, self.t_start + deliver)
            mid = due - self.first_due - STEP / 2                   # the window's middle, relative to the first window
            line = f"G,{self.n + 1},{(self.ticks0 + self.n * TICKS_PER_STEP) % (1 << 32)},{self.predict(mid)},3800\r\n"
            self.n += 1
            return line.encode()
        self.clock.t += self.timeout
        return b""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def performed(schedule):
    """predict function of a perfect board: the target of the block that holds the time, 'rest' in a pause."""
    def f(t):
        for b in schedule:
            if b["start_s"] <= t < b["end_s"]:
                return b["target"]
        return "rest"
    return f


def expected_scored(schedule, lead=la.LEAD_IN_S, tail=la.TAIL_GUARD_S):
    """Brute force: {block index: windows whose every second is inside the block's scored zone}, for windows ending at 0, 1.28, 2.56 ... s."""
    out = {b["index"]: 0 for b in schedule}
    k, horizon = 0, schedule[-1]["end_s"] + 5
    while k * STEP < horizon:
        end = k * STEP
        for b in schedule:
            if end - la.WINDOW_S >= b["start_s"] + lead and end <= b["end_s"] - tail:
                out[b["index"]] += 1
        k += 1
    return out


def synthetic_run(session="live1", seed=1, rounds=3, wrong=None, model_sha="ab" * 32, trained=("s1", "s2", "s3"), block_s=20.0, pause_s=8.0):
    """(meta, windows) of a finished run of a board that is right except where `wrong` ({block index: gesture it says instead}) says otherwise."""
    schedule = la.make_schedule(la.LABELS, rounds, seed, block_s, pause_s)
    wrong = wrong or {}
    windows, k = [], 0
    while k * STEP < schedule[-1]["end_s"] + 3:
        mid = k * STEP - STEP / 2
        b = next((b for b in schedule if b["start_s"] <= mid < b["end_s"]), None)
        pred = "rest" if b is None else wrong.get(b["index"], b["target"])
        windows.append({"t_s": k * STEP, "window": k + 1, "t_ticks": (1_000_000 + k * TICKS_PER_STEP) % (1 << 32), "predicted": pred, "compute_us": 3800})
        k += 1
    meta = {"session": session, "started_at": "2026-10-15T12:00:00", "seed": seed, "labels": list(la.LABELS), "rounds": rounds, "block_s": block_s,
            "pause_s": pause_s, "lead_in_s": la.LEAD_IN_S, "tail_guard_s": la.TAIL_GUARD_S, "schedule": schedule, "completed_s": schedule[-1]["end_s"],
            "interrupted": False, "monitor_problems": [], "model": {"file": "gesture_model.c", "sha256": model_sha, "trained_on": {"sessions": list(trained), "windows": 465}}}
    return meta, windows


# ---------------------------------------------------------------- READMEs with the old "not measured" wording (the sentences the repository had before the live test)
ORIGINAL_FILES = {
    "firmware/sparkfun_edge/gesture_demo/README.md": (
        "# Gesture chain for the SparkFun Edge: runs on the board, accuracy not yet measured\n\nintro line\n\n"
        "**Status: checked on the PC. The model was trained on every\nrecording, so it has no held-out accuracy; the on-device accuracy will come from a session "
        "recorded after the model was trained.**\n\n## Measured\ntext\n\n## Not done yet\n- A long run: not measured.\n"
        "- The live accuracy: a new session, recorded after the model was trained, with the gestures done in a known order. One person, one board.\n"),
    "firmware/sparkfun_edge/gesture_features/README.md": (
        "# Features\n\ntext\n\n## Not done yet\nFilling the 128-sample window from the sensor on the board, calling the tree, flashing it, and the on-device accuracy check on a\n"
        "held-out session.\n"),
    "firmware/sparkfun_edge/README.md": "# Board\n\n## Not done yet\n* The sample rate was not measured.\n* No machine learning has run on this board yet.\n",
    "README.md": "# Lab\n\n## Roadmap\n1. A thing.\n4. Check on-device accuracy against the PC on a held-out set.\n5. Another thing.\n",
    "firmware/sparkfun_edge/gesture_demo/gesture_model.json": '{"model": "forest", "trees": 100}\n',
}


def make_root(tmp_path, files=None):
    """A folder laid out like the repository, holding the READMEs as they were before the live test. Returns its path."""
    for rel, text in (files or ORIGINAL_FILES).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def two_runs(**kw):
    """Two finished runs of a board that is wrong in two blocks of the first run (guaranteed wrong: it names the next gesture in the list)."""
    sched = la.make_schedule(la.LABELS, 3, 1, 20.0, 8.0)
    wrong = {i: la.LABELS[(la.LABELS.index(sched[i]["target"]) + 1) % len(la.LABELS)] for i in (2, 6)}
    return [synthetic_run("live1", seed=1, wrong=wrong, **kw), synthetic_run("live2", seed=2, **kw)]
