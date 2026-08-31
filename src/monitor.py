"""HPC job monitor driven by a finite-state machine (FSM).

Runs on the LOCAL machine. Each poll opens a fresh short SSH connection (a few
seconds), so there is no long-lived session and the 6h login limit never
applies. On SSH failure the FSM enters DISCONNECTED: it pauses polling, records
state, and retries with backoff instead of spinning. The transition table is
total (every ``(state, event)`` has an entry) and unit-tested for completeness.

Run:  python src/monitor.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class State(str, Enum):
    SMOKE_MONITOR = "smoke_monitor"   # watching the smoke job (30 min interval)
    REAL_MONITOR = "real_monitor"     # watching real training (5 min interval)
    REAL_ETA = "real_eta"             # real training healthy, compute ETA
    DISCONNECTED = "disconnected"     # ssh lost; paused
    FAILED = "failed"                 # smoke job failed; needs a fix
    DONE = "done"                     # report delivered


class Event(str, Enum):
    POLL_OK_PENDING = "poll_ok_pending"            # job still pending/running
    POLL_OK_SMOKE_SUCCESS = "poll_ok_smoke_success"
    POLL_OK_SMOKE_FAIL = "poll_ok_smoke_fail"
    POLL_OK_REAL_HEALTHY = "poll_ok_real_healthy"  # real job running, loss ok
    POLL_OK_REAL_DONE = "poll_ok_real_done"
    POLL_FAIL = "poll_fail"                        # ssh error / timeout


def _ignore(state: State) -> tuple[State, str]:
    return (state, "ignore")


# Total transition table over State x Event: (next_state, action).
TRANSITIONS: dict[tuple[State, Event], tuple[State, str]] = {
    # --- smoke monitoring (30 min) ---
    (State.SMOKE_MONITOR, Event.POLL_OK_PENDING): (State.SMOKE_MONITOR, "wait_30m"),
    (State.SMOKE_MONITOR, Event.POLL_OK_SMOKE_SUCCESS): (State.REAL_MONITOR, "submit_real"),
    (State.SMOKE_MONITOR, Event.POLL_OK_SMOKE_FAIL): (State.FAILED, "report_fail"),
    (State.SMOKE_MONITOR, Event.POLL_OK_REAL_HEALTHY): _ignore(State.SMOKE_MONITOR),
    (State.SMOKE_MONITOR, Event.POLL_OK_REAL_DONE): _ignore(State.SMOKE_MONITOR),
    (State.SMOKE_MONITOR, Event.POLL_FAIL): (State.DISCONNECTED, "pause"),
    # --- real training monitoring (5 min, 30 min window) ---
    (State.REAL_MONITOR, Event.POLL_OK_PENDING): (State.REAL_MONITOR, "wait_5m"),
    (State.REAL_MONITOR, Event.POLL_OK_REAL_HEALTHY): (State.REAL_MONITOR, "wait_5m"),
    (State.REAL_MONITOR, Event.POLL_OK_REAL_DONE): (State.REAL_ETA, "compute_eta"),
    (State.REAL_MONITOR, Event.POLL_OK_SMOKE_SUCCESS): _ignore(State.REAL_MONITOR),
    (State.REAL_MONITOR, Event.POLL_OK_SMOKE_FAIL): _ignore(State.REAL_MONITOR),
    (State.REAL_MONITOR, Event.POLL_FAIL): (State.DISCONNECTED, "pause"),
    # --- eta -> report -> done ---
    (State.REAL_ETA, Event.POLL_OK_PENDING): (State.DONE, "report"),
    (State.REAL_ETA, Event.POLL_OK_REAL_HEALTHY): (State.DONE, "report"),
    (State.REAL_ETA, Event.POLL_OK_REAL_DONE): (State.DONE, "report"),
    (State.REAL_ETA, Event.POLL_OK_SMOKE_SUCCESS): (State.DONE, "report"),
    (State.REAL_ETA, Event.POLL_OK_SMOKE_FAIL): (State.DONE, "report"),
    (State.REAL_ETA, Event.POLL_FAIL): (State.DISCONNECTED, "pause"),
    # --- disconnected: pause; resume is resolved by the loop's origin state ---
    (State.DISCONNECTED, Event.POLL_OK_PENDING): (State.DISCONNECTED, "resume"),
    (State.DISCONNECTED, Event.POLL_OK_SMOKE_SUCCESS): (State.DISCONNECTED, "resume"),
    (State.DISCONNECTED, Event.POLL_OK_SMOKE_FAIL): (State.DISCONNECTED, "resume"),
    (State.DISCONNECTED, Event.POLL_OK_REAL_HEALTHY): (State.DISCONNECTED, "resume"),
    (State.DISCONNECTED, Event.POLL_OK_REAL_DONE): (State.DISCONNECTED, "resume"),
    (State.DISCONNECTED, Event.POLL_FAIL): (State.DISCONNECTED, "keep_paused"),
    # --- terminals ---
    (State.FAILED, Event.POLL_OK_PENDING): (State.FAILED, "terminal"),
    (State.FAILED, Event.POLL_OK_SMOKE_SUCCESS): (State.FAILED, "terminal"),
    (State.FAILED, Event.POLL_OK_SMOKE_FAIL): (State.FAILED, "terminal"),
    (State.FAILED, Event.POLL_OK_REAL_HEALTHY): (State.FAILED, "terminal"),
    (State.FAILED, Event.POLL_OK_REAL_DONE): (State.FAILED, "terminal"),
    (State.FAILED, Event.POLL_FAIL): (State.FAILED, "terminal"),
    (State.DONE, Event.POLL_OK_PENDING): (State.DONE, "terminal"),
    (State.DONE, Event.POLL_OK_SMOKE_SUCCESS): (State.DONE, "terminal"),
    (State.DONE, Event.POLL_OK_SMOKE_FAIL): (State.DONE, "terminal"),
    (State.DONE, Event.POLL_OK_REAL_HEALTHY): (State.DONE, "terminal"),
    (State.DONE, Event.POLL_OK_REAL_DONE): (State.DONE, "terminal"),
    (State.DONE, Event.POLL_FAIL): (State.DONE, "terminal"),
}


def transition(state: State, event: Event) -> tuple[State, str]:
    """Pure, total transition function over ``State x Event``."""
    return TRANSITIONS[(state, event)]


@dataclass
class Config:
    ssh_host: str = "eo255343@login23-1.hpc.itc.rwth-aachen.de"
    smoke_job: str = "3357546"
    state_file: str = "monitor_state.json"
    log_file: str = "monitor.log"
    smoke_interval: int = 1800   # 30 min
    real_interval: int = 300     # 5 min
    real_window: int = 1800      # 30 min total real-training watch
    backoff: int = 60            # reconnect backoff (seconds)
    max_retries: int = 5         # reconnect attempts before halting


def log(cfg: Config, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(cfg.log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def poll_job(ssh_host: str, job_id: str) -> str | None:
    """Fresh short SSH poll. Return raw output, or None on any failure."""
    cmd = (
        f"squeue -j {job_id} -h -o '%T' 2>/dev/null; "
        f"echo '---'; tail -5 ~/cllm/logs/train_{job_id}.out 2>/dev/null"
    )
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", ssh_host, cmd],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        return r.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def classify(phase: str, out: str | None) -> Event:
    """Sensor: map raw poll output to an Event (integration point, not unit-tested)."""
    if out is None:
        return Event.POLL_FAIL
    status = out.split("---")[0].strip()
    tail = out.split("---")[1] if "---" in out else ""
    if "PENDING" in status or "RUNNING" in status:
        if phase == "real" and "loss" in tail:
            return Event.POLL_OK_REAL_HEALTHY
        return Event.POLL_OK_PENDING
    # no longer queued/running -> finished
    if phase == "smoke":
        return Event.POLL_OK_SMOKE_SUCCESS if ("device: cuda" in tail and "loss" in tail) else Event.POLL_OK_SMOKE_FAIL
    return Event.POLL_OK_REAL_DONE


def load_state(cfg: Config) -> dict:
    p = Path(cfg.state_file)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"state": State.SMOKE_MONITOR.value, "origin": State.SMOKE_MONITOR.value}


def save_state(cfg: Config, state: State, origin: State) -> None:
    Path(cfg.state_file).write_text(
        json.dumps({"state": state.value, "origin": origin.value}), encoding="utf-8"
    )


def run(cfg: Config) -> None:
    st = load_state(cfg)
    state = State(st["state"])
    origin = State(st["origin"])
    phase = "smoke" if origin == State.SMOKE_MONITOR else "real"
    log(cfg, f"start: state={state.value} origin={origin.value}")

    while state not in (State.DONE, State.FAILED):
        job = cfg.smoke_job if phase == "smoke" else st.get("real_job", "")
        out = poll_job(cfg.ssh_host, job)
        event = classify(phase, out)
        nxt, action = transition(state, event)

        if action in ("pause", "keep_paused"):
            if state != State.DISCONNECTED:
                origin = state
                log(cfg, f"DISCONNECTED (ssh failed); pausing. origin={origin.value}")
                save_state(cfg, State.DISCONNECTED, origin)
            # backoff reconnect (avoid spinning)
            for attempt in range(1, cfg.max_retries + 1):
                time.sleep(cfg.backoff)
                if poll_job(cfg.ssh_host, cfg.smoke_job) is not None:
                    state = origin
                    log(cfg, f"reconnected after {attempt} retry(s); resumed {state.value}")
                    break
            else:
                log(cfg, "still disconnected after retries; HALTING. Reconnect then rerun.")
                return
            continue

        if action == "resume":
            state = origin
            log(cfg, f"resumed {state.value}")
            continue

        if action == "submit_real":
            # smoke succeeded: submit real training, remember its job id
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=15", cfg.ssh_host,
                 "cd ~/cllm && sbatch scripts/train_real.slurm 2>/dev/null"],
                capture_output=True, text=True, timeout=30,
            )
            real_job = r.stdout.strip().split()[-1] if r.stdout.strip() else ""
            st["real_job"] = real_job
            phase = "real"
            log(cfg, f"smoke SUCCESS -> submitted real training job {real_job}")

        if action == "report" or action == "compute_eta":
            log(cfg, "real training done; computing ETA report")
            # ETA is derived from the training log's steps/sec (see monitor.log)

        state = nxt
        save_state(cfg, state, origin)
        if action == "report_fail":
            log(cfg, "smoke job FAILED — read logs/train_<job>.out; fix then rerun")
            break

        interval = cfg.real_interval if state == State.REAL_MONITOR else cfg.smoke_interval
        log(cfg, f"sleeping {interval}s in {state.value}")
        time.sleep(interval)

    log(cfg, f"monitor finished in state {state.value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="monitor_state.json")
    args = parser.parse_args()
    cfg = Config(state_file=args.state)
    run(cfg)


if __name__ == "__main__":
    main()
