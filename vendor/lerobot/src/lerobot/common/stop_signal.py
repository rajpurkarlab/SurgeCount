"""Stop-window detection in decoded action-target coordinates."""
import math


def first_stop_window(values, window: int, threshold: float = 0.5):
    """Return the first index of a consecutive-high window, or None.

    Values must be decoded with the action postprocessor before use.
    Windows are evaluated within one predicted chunk.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("Stop window must be a positive integer")
    count = 0
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError("Stop predictions must be finite")
        count = count + 1 if value > threshold else 0
        if count >= window:
            return index - window + 1
    return None
