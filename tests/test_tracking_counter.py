"""Synthetic checks of the counter-attack indicator of amendment G (research/dq/tracking_counter.py): runners, back, goalkeepers, the position-only proxy."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "dq"))
import tracking_counter as tc


def _frame(n_a=11, n_b=11):
    is_a = np.array([True] * n_a + [False] * n_b); gk = np.zeros(n_a + n_b, bool); gk[0] = True; gk[n_a] = True
    return is_a, gk


def test_three_runners_against_one_defender_is_an_advantage():
    is_a, gk = _frame(); xs = np.full(22, np.nan); vx = np.zeros(22)
    xs[0] = -50; xs[1] = -30                                              # A: keeper and one defender goal-side of the ball (x = -10); carrier = index 2 at the ball
    xs[2] = -10; xs[3:11] = 10                                            # the other A players are ahead of the ball
    xs[11] = 50; xs[12:15] = [-12, -14, -16]; vx[12:15] = -5.0            # B: keeper, three runners moving toward A's goal, ahead of or level with the ball
    xs[15:22] = 5; vx[15:22] = 0.0
    run, back, adv = tc.counter_indicator(-10.0, xs, vx, is_a, gk, carrier=2)
    assert run == 3 and back == 2 and adv == 1                            # keeper + one defender behind the ball, the carrier excluded
    xs[1] = 0.0                                                           # the defender is no longer goal-side of the ball
    assert tc.counter_indicator(-10.0, xs, vx, is_a, gk, carrier=2) == (3, 1, 2)


def test_the_opposing_keeper_and_slow_players_are_not_runners():
    is_a, gk = _frame(); xs = np.full(22, 0.0); vx = np.zeros(22); vx[11] = -9.0; vx[12] = -2.0     # B keeper (index 11) moving fast, one B player too slow
    assert tc.counter_indicator(5.0, xs, vx, is_a, gk, carrier=1)[0] == 0
    vx[13] = -2.5; assert tc.counter_indicator(5.0, xs, vx, is_a, gk, carrier=1)[0] == 1


def test_players_far_behind_the_ball_line_do_not_count_as_runners():
    is_a, gk = _frame(); xs = np.full(22, 0.0); vx = np.zeros(22); xs[12] = 20.0; vx[12] = -6.0       # 20 m behind a ball at x = 0 (more than the 5 m margin)
    assert tc.counter_indicator(0.0, xs, vx, is_a, gk, carrier=1)[0] == 0
    xs[12] = 4.0; assert tc.counter_indicator(0.0, xs, vx, is_a, gk, carrier=1)[0] == 1


def test_untracked_players_are_ignored():
    is_a, gk = _frame(); xs = np.full(22, np.nan); vx = np.full(22, np.nan)
    assert tc.counter_indicator(0.0, xs, vx, is_a, gk, carrier=-1) == (0, 0, 0)


def test_proxy_uses_positions_only_and_the_visible_subset():
    is_a, _ = _frame(); xs = np.full(22, 10.0); xs[:3] = [-30, -20, -5]; xs[11:14] = [-25, -15, -8]         # three A and three B players goal-side of a ball at x = 0
    keep = np.ones(22, bool); assert tc.proxy_adv(0.0, xs, is_a, carrier=2, keep=keep) == 3 - 2             # carrier (index 2) excluded on the A side
    keep[11:14] = False; assert tc.proxy_adv(0.0, xs, is_a, carrier=2, keep=keep) == -2                     # hidden B players are not counted
