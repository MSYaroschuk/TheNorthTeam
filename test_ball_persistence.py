#!/usr/bin/env python3
# test_ball_persistence.py
# ponytail: mirrors the confirm-then-remember gate in auto_chassis.py.
# Standalone on purpose - importing auto_chassis constructs Servo objects
# at module level, which would touch the HAT.

from dataclasses import dataclass

BALL_CONFIRM_FRAMES = 2
BALL_MEMORY_S = 0.4
BALL_MATCH_PX = 80


@dataclass
class Ball:
    x: int
    y: int


class Gate:
    """The state machine from auto_chassis.py lines 231-251."""

    def __init__(self):
        self.confirmed_ball = None
        self.confirmed_at = 0.0
        self.hit_streak = 0

    def step(self, raw_ball, now):
        """One frame. Returns the ball to act on, or None."""
        if raw_ball is not None:
            same_ball = (
                self.confirmed_ball is not None
                and abs(raw_ball.x - self.confirmed_ball.x) <= BALL_MATCH_PX
                and abs(raw_ball.y - self.confirmed_ball.y) <= BALL_MATCH_PX
            )
            self.hit_streak = self.hit_streak + 1 if same_ball else 1
            self.confirmed_ball = raw_ball
            self.confirmed_at = now
        elif self.hit_streak >= BALL_CONFIRM_FRAMES:
            # already trusted: coast through a short dropout
            if (now - self.confirmed_at) > BALL_MEMORY_S:
                self.confirmed_ball = None
                self.hit_streak = 0
        else:
            # unconfirmed candidate: a miss breaks the streak at once
            self.confirmed_ball = None
            self.hit_streak = 0

        return self.confirmed_ball if self.hit_streak >= BALL_CONFIRM_FRAMES else None


def test_single_frame_phantom_is_ignored():
    """A false positive that appears for one frame must never be acted on."""
    g = Gate()
    assert g.step(Ball(300, 200), 0.0) is None, "acted on a 1-frame phantom"
    assert g.step(None, 0.14) is None
    assert g.step(None, 0.28) is None


def test_real_ball_confirmed_on_second_frame():
    """Two consecutive hits near each other = real, act on it."""
    g = Gate()
    assert g.step(Ball(300, 200), 0.0) is None, "acted before confirmation"
    got = g.step(Ball(305, 210), 0.14)
    assert got is not None, "failed to confirm a steady ball"
    assert got.x == 305


def test_one_dropped_frame_does_not_lose_the_ball():
    """The stutter fix: a gap shorter than memory keeps the ball tracked."""
    g = Gate()
    g.step(Ball(300, 200), 0.0)
    assert g.step(Ball(300, 200), 0.14) is not None
    got = g.step(None, 0.28)          # dropped frame, 0.14s later
    assert got is not None, "braked on a single dropped frame"
    assert got.x == 300, "forgot the remembered position"


def test_memory_expires_when_ball_really_gone():
    """A gap longer than BALL_MEMORY_S must release the ball."""
    g = Gate()
    g.step(Ball(300, 200), 0.0)
    assert g.step(Ball(300, 200), 0.14) is not None
    assert g.step(None, 0.60) is None, "held a ball past the memory window"


def test_jumping_phantom_restarts_streak():
    """Detections far apart are different objects, not a confirmation."""
    g = Gate()
    assert g.step(Ball(50, 100), 0.0) is None
    assert g.step(Ball(600, 400), 0.14) is None, "confirmed two unrelated blobs as one ball"


def test_flicker_never_confirms():
    """REGRESSION: alternating hit/miss must never accumulate a confirmation.

    The first version let hit_streak survive a dropout even when the ball was
    not yet confirmed, so two phantoms BALL_MEMORY_S apart confirmed each
    other and 'consecutive' meant nothing. This is the bug that let false
    positives through.
    """
    g = Gate()
    now = 0.0
    for _ in range(8):
        assert g.step(Ball(300, 200), now) is None, "flicker produced a lock"
        now += 0.14
        assert g.step(None, now) is None, "flicker produced a lock on the gap"
        now += 0.14
    assert g.hit_streak < BALL_CONFIRM_FRAMES


def test_two_phantoms_across_a_gap_do_not_confirm():
    """REGRESSION: a phantom, a dropped frame, then a second phantom nearby.

    Under the old logic confirmed_ball survived the gap, so the second hit
    matched it and the streak reached 2 -> the robot chased a phantom.
    """
    g = Gate()
    assert g.step(Ball(300, 200), 0.00) is None
    assert g.step(None, 0.14) is None                 # gap, still under memory
    got = g.step(Ball(310, 205), 0.28)                # nearby phantom
    assert got is None, "two phantoms across a gap confirmed each other"


def test_confirmed_ball_still_coasts_after_the_fix():
    """The stutter fix must survive the stricter streak rule."""
    g = Gate()
    g.step(Ball(300, 200), 0.00)
    assert g.step(Ball(302, 203), 0.14) is not None    # confirmed
    assert g.step(None, 0.28) is not None, "confirmed ball lost on one gap"
    assert g.step(None, 0.40) is not None, "confirmed ball lost inside memory"


def test_match_radius_rejects_a_jump_of_100px():
    """80px radius: a 100px jump is a different object, not the same ball."""
    g = Gate()
    assert g.step(Ball(200, 200), 0.00) is None
    assert g.step(Ball(300, 200), 0.14) is None, "100px jump treated as same ball"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
