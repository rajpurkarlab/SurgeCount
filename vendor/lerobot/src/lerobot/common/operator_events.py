"""Synchronize keyboard intent with asynchronous planner responses."""
from threading import RLock


def operator_lock(events):
    return events.setdefault("_operator_lock", RLock())


def operator_activity(events, preserve_pending=False):
    """Invalidate older planner requests for every received operator input."""
    with operator_lock(events):
        events["operator_revision"] = events.get("operator_revision", 0) + 1
        if not preserve_pending:
            events.pop("pending_planner_decision", None)


def _planner_ready(events):
    return events.get("current_task") is None and not any(events.get(key) for key in (
        "exit_early", "stop_recording", "rerecord_episode", "go_home",
        "operator_slot_pending", "pending_planner_decision",
    ))


def planner_snapshot(events):
    """Return the operator revision for a new request, or None while busy."""
    with operator_lock(events):
        if _planner_ready(events):
            return events.get("operator_revision", 0)
    return None


def planner_result_current(events, revision):
    with operator_lock(events):
        return revision == events.get("operator_revision", 0) and _planner_ready(events)


def apply_planner_result(events, revision, label, pending=None):
    """Atomically check intent and install a result; keyboard changes always win."""
    with operator_lock(events):
        if not planner_result_current(events, revision):
            return False
        if pending is None:
            events["current_task"] = label
        else:
            events["pending_planner_decision"] = pending
        return True
