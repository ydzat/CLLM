"""FSM completeness tests for the monitor (no HPC needed)."""
from collections import deque

from src.monitor import Event, State, TRANSITIONS, transition

TERMINAL = {State.FAILED, State.DONE}


def test_transition_table_is_total():
    # every (state, event) has a defined transition
    for s in State:
        for e in Event:
            assert (s, e) in TRANSITIONS, f"missing transition for ({s.value}, {e.value})"


def test_all_states_reachable():
    seen = {State.SMOKE_MONITOR}
    q = deque([State.SMOKE_MONITOR])
    while q:
        s = q.popleft()
        for e in Event:
            nxt, _ = transition(s, e)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == set(State), f"unreachable states: {set(State) - seen}"


def test_no_deadlock():
    # every non-terminal state has at least one outgoing transition
    for s in State:
        if s not in TERMINAL:
            outs = {transition(s, e)[0] for e in Event}
            assert len(outs) >= 1, f"deadlock at {s.value}"


def test_smoke_success_goes_to_real():
    assert transition(State.SMOKE_MONITOR, Event.POLL_OK_SMOKE_SUCCESS) == (State.REAL_MONITOR, "submit_real")


def test_smoke_fail_goes_to_failed():
    assert transition(State.SMOKE_MONITOR, Event.POLL_OK_SMOKE_FAIL) == (State.FAILED, "report_fail")


def test_poll_fail_goes_to_disconnected():
    assert transition(State.SMOKE_MONITOR, Event.POLL_FAIL) == (State.DISCONNECTED, "pause")
    assert transition(State.REAL_MONITOR, Event.POLL_FAIL) == (State.DISCONNECTED, "pause")


def test_disconnected_is_not_terminal_deadlock():
    # DISCONNECTED can resume or stay paused — both meaningful, no deadlock
    assert transition(State.DISCONNECTED, Event.POLL_OK_PENDING) == (State.DISCONNECTED, "resume")
    assert transition(State.DISCONNECTED, Event.POLL_FAIL) == (State.DISCONNECTED, "keep_paused")


def test_terminals_absorb_all_events():
    for s in TERMINAL:
        for e in Event:
            nxt, action = transition(s, e)
            assert action == "terminal"
            assert nxt == s
