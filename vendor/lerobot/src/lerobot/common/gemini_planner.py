"""Gemini API high-level planner for the bimanual gauze pi0.5 demo.

Mirrors `claude_planner.py` but uses Google's google-genai SDK and a single
forward pass with JSON-schema-constrained output (no self-check second pass).
Picks one of 12 skill keys at each freeze boundary based on the 3 camera
images and speaks the rationale aloud through `log_say` so the operator hears
what the model decided.

Lazy-imports google.genai + Pillow + numpy so the package stays optional.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.common.planner_types import (
    SKILL_DESCRIPTIONS,
    SKILL_DESCRIPTIONS_MINIMAL,
    SKILL_ENUM,
    SLOT_NUMBERS,
    PlannerDecision,
)
from lerobot.utils.utils import log_say
from lerobot.common.operator_events import apply_planner_result, planner_result_current, planner_snapshot

# Short spoken form for each skill, read aloud via spd-say so the operator
# can hear which skill the planner chose without reading the console.
SKILL_VOICE: dict[str, str] = {
    "1":    "right arm grab blue",
    "2a":   "place into slot one",
    "2b":   "place into slot two",
    "2c":   "place into slot three",
    "3":    "right arm release",
    "4":    "right arm back",
    "5":    "left arm grab edge",
    "6":    "right arm grab exposed blue",
    "7":    "left arm release",
    "8":    "left arm drop and home",
    "9":    "left arm back",
    "ra_home": "right arm hardcoded home",
    "wait": "waiting",
}

LOGGER = logging.getLogger(__name__)


def _slots_to_list(s) -> list[int]:
    return sorted(int(x) for x in (s or set()))


class GeminiPlanner:
    """High-level planner: 3 images -> one of 12 pi0.5 skill strings.

    Same interface as ClaudePlanner so lerobot_record.py can swap between them:
        planner = GeminiPlanner(SUBTASK_LABELS, model="gemini-3.1-flash-lite", ...)
        planner.reset_history()
        planner.maybe_call(events, obs, single_task)

    Mutates events["current_task"] when (and only when) it picks a non-wait
    skill AND dry_run is False. Speaks the chosen skill + rationale through
    log_say on every successful non-wait pick.
    """

    def __init__(
        self,
        subtask_labels: dict[str, str],
        *,
        dry_run: bool = False,
        cooldown_s: float = 1.0,
        model: str = "gemini-3.1-flash-lite",
        speak: bool = True,
        log_dir: str | Path | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        confirm_required: bool = False,
    ):
        # Lazy imports keep optional deps actually optional.
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "GeminiPlanner requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from e
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "GeminiPlanner requires Pillow for JPEG encoding. "
                "Install with: pip install pillow"
            ) from e
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "GeminiPlanner requires numpy. Install with: pip install numpy"
            ) from e

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"GeminiPlanner: environment variable {api_key_env} is not set. "
                "Export it before launching the eval script."
            )

        self._genai = genai
        self._types = genai_types
        self._Image = Image
        self._np = np
        self._client = genai.Client(api_key=api_key)

        self.subtask_labels = dict(subtask_labels)
        self.dry_run = dry_run
        self.cooldown_s = cooldown_s
        self.model = model
        self.speak = speak
        # Test mode: when True, decisions are written to
        # events["pending_planner_decision"] instead of events["current_task"].
        # The operator must press Space to confirm (promotes pending into
        # current_task) or any 1-9 key to override. See control_utils.py.
        self.confirm_required = confirm_required

        self._last_call_t: float = 0.0
        self._last_chosen_key: str | None = None
        self._last_blue_confidence: float | None = None
        self._labeled_history: deque[dict] = deque(maxlen=20)
        # Track consecutive wrong-grab events: a wrong grab is detected when the
        # planner picks "4" (RA back) immediately after "1" (grab from tray) or
        # "6" (grab exposed blue) — that combo means the gauze wasn't held by
        # the blue thread and the arm is returning home. Reset on a successful
        # "3" (release into slot) or on episode reset. After
        # _WRONG_GRAB_LIMIT consecutive wrong-grabs we tell the model to
        # switch strategy.
        self._wrong_grab_count: int = 0
        self._WRONG_GRAB_LIMIT: int = int(os.environ.get("WRONG_GRAB_LIMIT", "3"))
        # Multi-turn chat handle — lazily created on first call, reset per
        # episode so each episode is one continuous conversation that
        # accumulates context across decisions.
        self._chat = None

        self._log_path: Path | None = None
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._log_path = log_dir / f"gemini_planner_{ts}.jsonl"
            LOGGER.info("GeminiPlanner JSONL trace -> %s", self._log_path)

        self._system_text = self._build_system_prompt()
        self._response_schema = self._build_response_schema()

        LOGGER.info(
            "GeminiPlanner ready (model=%s, dry_run=%s, speak=%s, cooldown_s=%.1f)",
            model, dry_run, speak, cooldown_s,
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        """Call at episode start. Clears last-chosen, cooldown, chat
        history, wrong-grab streak, and operator-labeled outcome history
        so each new episode is its own conversation with no carryover."""
        self._last_chosen_key = None
        self._last_blue_confidence = None
        self._last_call_t = 0.0
        self._wrong_grab_count = 0
        self._chat = None
        self._labeled_history.clear()

    def last_blue_confidence(self) -> float | None:
        return self._last_blue_confidence

    # ------------------------------------------------------------------
    # Main entry point — called from lerobot_record.py freeze block
    # ------------------------------------------------------------------
    def maybe_call(
        self,
        events: dict,
        obs: dict[str, Any],
        single_task: str,
    ) -> PlannerDecision | None:
        operator_revision = planner_snapshot(events)
        if operator_revision is None:
            return None

        outcome = events.pop("last_subtask_outcome", None)
        if outcome is not None and self._last_chosen_key is not None:
            self._labeled_history.append({
                "chosen_key": self._last_chosen_key,
                "outcome": outcome.get("outcome", "unknown"),
                "note": outcome.get("note"),
            })
            LOGGER.info(
                "GeminiPlanner: previous %s labeled %s",
                self._last_chosen_key, outcome.get("outcome"),
            )

        now = time.perf_counter()
        if now - self._last_call_t < self.cooldown_s:
            return None

        cam_keys = {
            "top":         "left_top",
            "left_wrist":  "left_left_front",
            "right_wrist": "right_right_front",
        }
        images: dict[str, Any] = {}
        for label, k in cam_keys.items():
            arr = obs.get(k)
            if arr is None:
                LOGGER.warning(
                    "GeminiPlanner: observation missing camera key %r; skipping call", k,
                )
                return None
            images[label] = arr

        slots_filled = _slots_to_list(events.get("slots_filled"))
        # Two possible context flags:
        #  - timeout_recovery_context: post-recovery (RA already moved home),
        #    asking the model to adjust strategy for the NEXT subtask.
        #  - timeout_inspection_pending: timeout just fired and BOTH arms are
        #    frozen at the timeout pose; the model must inspect grippers and
        #    pick an immediate next action (extend, RA drop, LA drop, etc.).
        timeout_ctx = events.pop("timeout_recovery_context", None)
        inspect_ctx = events.pop("timeout_inspection_pending", None)

        self._last_call_t = now
        t0 = time.perf_counter()
        try:
            decision = self._call_once(images, slots_filled, timeout_ctx=timeout_ctx, inspect_ctx=inspect_ctx)
        except Exception as e:
            LOGGER.exception("GeminiPlanner call failed: %s", e)
            decision = PlannerDecision(
                chosen_key="wait",
                rationale=f"API error: {e}",
                visible_blue_confidence=0.0,
                thinking_summary="",
                latency_s=0.0,
                label=None,
                self_check_revised=False,
            )

        decision.latency_s = time.perf_counter() - t0
        decision.label = self._resolve_label(decision.chosen_key)
        if not planner_result_current(events, operator_revision):
            LOGGER.info("Discarding planner response because operator intent changed during the request")
            return None
        # Wrong-grab tracking. Increments on a signal that skill 1 or 6 FAILED:
        #   (4 or ra_home) after (1 or 6): RA recovery => missed/empty/wrong grab.
        # NOTE: 6 -> 7 is NOT a failure — for Strategy B, grabbing ANY part of
        # the gauze is success, so 6 -> 7 -> 9 -> 2x is the success handoff.
        # The only skill-6 failure is an empty gripper (6 -> ra_home).
        # Resets on:
        #   3: success (gauze placed into slot).
        #   8: full LA abort (clearest strategy-switch signal).
        #   any pick outside the grab/recovery cluster {1,4,5,6,7,9,ra_home}:
        #       genuine strategy switch (e.g. 2x for placing). 5/7/9 are kept in
        #       the cluster so a missed-grab streak (6 -> ra_home -> 6 ...)
        #       accumulates across retries.
        _is_wrong_grab_signal = (
            decision.chosen_key in ("4", "ra_home") and self._last_chosen_key in ("1", "6")
        )
        if _is_wrong_grab_signal:
            self._wrong_grab_count += 1
            LOGGER.info(
                "GeminiPlanner: wrong-grab #%d (%s -> %s)",
                self._wrong_grab_count, self._last_chosen_key, decision.chosen_key,
            )
        elif decision.chosen_key in ("3", "8"):
            if self._wrong_grab_count > 0:
                LOGGER.info(
                    "GeminiPlanner: wrong-grab counter reset (was %d) — %s",
                    self._wrong_grab_count,
                    "success (skill 3)" if decision.chosen_key == "3" else "full LA abort (skill 8)",
                )
            self._wrong_grab_count = 0
        elif decision.chosen_key not in ("1", "4", "5", "6", "7", "9", "ra_home"):
            # Pick is outside the grab/recovery cluster (e.g. 2x placing).
            # Treat as a strategy change and reset.
            if self._wrong_grab_count > 0:
                LOGGER.info(
                    "GeminiPlanner: wrong-grab counter reset (was %d) — strategy change (chosen=%s)",
                    self._wrong_grab_count, decision.chosen_key,
                )
            self._wrong_grab_count = 0
        self._last_chosen_key = decision.chosen_key
        if decision.rationale and not decision.rationale.startswith("API error"):
            self._last_blue_confidence = decision.visible_blue_confidence
        self._log_decision(decision, slots_filled)

        if not self.dry_run and decision.label is not None:
            voice = SKILL_VOICE.get(decision.chosen_key, decision.chosen_key)
            rationale_short = (decision.rationale or "").strip().rstrip(".")
            if self.confirm_required:
                # Test mode: park the decision in pending and wait for
                # operator confirmation (Space) or override (1-9 etc.).
                pending_decision = {
                    "chosen_key": decision.chosen_key,
                    "label": decision.label,
                    "rationale": decision.rationale,
                    "visible_blue_confidence": decision.visible_blue_confidence,
                }
                if not apply_planner_result(events, operator_revision, decision.label, pending_decision):
                    return None
                LOGGER.info(
                    "GeminiPlanner [PENDING] -> %s (%r) | conf=%.2f latency=%.1fs — press SPACE to confirm or 1-9 to override",
                    decision.chosen_key, decision.label,
                    decision.visible_blue_confidence, decision.latency_s,
                )
                if self.speak:
                    if rationale_short:
                        log_say(f"Gemini suggests {voice}. {rationale_short}. Press space to confirm.", play_sounds=True)
                    else:
                        log_say(f"Gemini suggests {voice}. Press space to confirm.", play_sounds=True)
            else:
                if not apply_planner_result(events, operator_revision, decision.label):
                    return None
                LOGGER.info(
                    "GeminiPlanner -> %s (%r) | conf=%.2f latency=%.1fs",
                    decision.chosen_key, decision.label,
                    decision.visible_blue_confidence, decision.latency_s,
                )
                if self.speak:
                    if rationale_short:
                        log_say(f"{voice}. {rationale_short}.", play_sounds=True)
                    else:
                        log_say(voice, play_sounds=True)
        elif self.dry_run:
            LOGGER.info(
                "[DRY-RUN] GeminiPlanner would pick %s | conf=%.2f latency=%.1fs | %s",
                decision.chosen_key, decision.visible_blue_confidence,
                decision.latency_s, decision.rationale,
            )
            # In dry-run / observe mode, still announce the predicted subtask
            # so the operator can hear what the model thinks the robot is
            # doing right now (without injecting into events["current_task"]).
            if self.speak and decision.label is not None:
                voice = SKILL_VOICE.get(decision.chosen_key, decision.chosen_key)
                rationale_short = (decision.rationale or "").strip().rstrip(".")
                if rationale_short:
                    log_say(f"observing {voice}. {rationale_short}.", play_sounds=True)
                else:
                    log_say(f"observing {voice}.", play_sounds=True)
        elif decision.label is None:
            LOGGER.info(
                "GeminiPlanner chose 'wait' | latency=%.1fs | %s",
                decision.latency_s, decision.rationale,
            )
            if self.speak and not decision.rationale.startswith("API error"):
                rationale_short = (decision.rationale or "").strip().rstrip(".")
                if rationale_short:
                    log_say(f"waiting. {rationale_short}.", play_sounds=True)

        return decision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_label(self, key: str) -> str | None:
        if key == "wait":
            return None
        if key in SLOT_NUMBERS:
            template = self.subtask_labels.get("2")
            if template is None:
                LOGGER.error("Subtask key '2' missing from SUBTASK_LABELS")
                return None
            return template.replace("{N}", str(SLOT_NUMBERS[key]))
        return self.subtask_labels.get(key)

    def _build_system_prompt(self) -> str:
        # GEMINI_PROMPT_STYLE controls how much guidance Gemini receives.
        # "detailed" (default): full strategy + visual gates + recovery rules.
        # "minimal": only goal + skill names + one-line descriptions, no
        # strategy guidance — used to test Gemini's intrinsic ability to plan.
        _style = os.environ.get("GEMINI_PROMPT_STYLE", "detailed").lower()
        if _style == "minimal":
            return self._build_system_prompt_minimal()
        return self._build_system_prompt_detailed()

    def _build_system_prompt_minimal(self) -> str:
        skills_table = "\n".join(
            f"  - {k}: {desc}" for k, desc in SKILL_DESCRIPTIONS_MINIMAL.items()
        )
        return f"""You are the high-level planner for a dual-arm sponge-sorting robot.

SCENE: A green tray holds up to 3 surgical gauzes. Each gauze is a white fabric pad with a blue thread sewn through it. Beside the tray sits a counting tray with 3 numbered slots (1, 2, 3). The robot has two arms (LA = left arm, RA = right arm). Three cameras are available each call:
  - TOP: overhead view of both trays.
  - LEFT_WRIST: viewpoint mounted on the left gripper.
  - RIGHT_WRIST: viewpoint mounted on the right gripper.

GOAL: Move gauzes from the green tray into the counting-tray slots, ONE AT A TIME. Each gauze must be grabbed by its BLUE THREAD with the right arm. Place each gauze into the NEXT empty slot of the counting tray, in order (slot 1 first, then 2, then 3). Do NOT attempt to carry more than one gauze at a time. When all 3 slots are filled, the task is complete and you may output 'wait'.

RETRIES: You may RETRY any subtask as many times as needed — there is NO retry limit. If a grab or placement fails, just attempt it again (or try a different subtask). Keep going until there is no gauze left in the green tray.

AVAILABLE SUBTASKS — pick exactly ONE of these enum values per decision:
{skills_table}

OUTPUT: Return ONLY a JSON object with these fields:
  - subtask_key: one of the enum values above.
  - rationale: one short sentence (<=200 chars) explaining your choice.
  - visible_blue_confidence: float in [0.0, 1.0], your confidence that the blue thread is currently visible somewhere in the scene.
"""

    def _build_system_prompt_detailed(self) -> str:
        skills_table = "\n".join(
            f"  - {k}: {desc}" for k, desc in SKILL_DESCRIPTIONS.items()
        )
        return f"""You are the high-level planner for a dual-arm surgical-sponge sorting robot.

SCENE
A green tray holds up to 3 white surgical gauzes. Each gauze has a blue thread sewn through the white fabric. Beside the tray sits a counting tray with 3 numbered slots (1, 2, 3). The robot has two arms (LA = left arm, RA = right arm). Three cameras are available each call:
  - TOP: overhead view of both trays.
  - LEFT_WRIST: viewpoint mounted on the left gripper.
  - RIGHT_WRIST: viewpoint mounted on the right gripper.

GOAL
For each gauze, place it in the next empty slot of the counting tray. Happy paths:
  Strategy A (blue visible in the RIGHT_WRIST camera):  1 -> 2x -> 3 -> 4
  Strategy B (blue NOT visible in right_wrist):         5 -> 6 -> 7 -> 9 -> 2x -> 3 -> 4
  where 2x = 2a/2b/2c per the next empty slot.

STRATEGY DECISION — use the RIGHT_WRIST camera ONLY to judge whether a blue thread is visible/grabbable on a gauze in the green tray. Ignore the top and left_wrist cameras for THIS decision (they may show blue that the RA cannot actually reach/grab). Pick Strategy A (1) ONLY if the blue thread is clearly visible in the RIGHT_WRIST view. If the right_wrist camera does not clearly show a grabbable blue thread, use Strategy B (5 -> LA lifts the white edge so blue dangles -> 6 RA grabs it), even if blue appears in the top camera.

VISUAL VERIFICATION GATES (all checks use right_wrist unless noted; pick 4 — or 8 for LA-side — if any check fails)
  After 1 (RA grab from tray):       Is the BLUE THREAD present in the RA gripper in ANY of these ways: (a) alone, (b) alongside extra gauze fabric, OR (c) PARTIALLY EMBEDDED inside the white fabric (only part of the blue is visible / the blue is woven through the white at the grip point)?
                                       yes -> 2x        no -> 4 (drop+return, retry gauze with 1 or 5)
                                     KEY POINT: ALL THREE of these count as a valid grasp:
                                       - blue thread alone in the grip,
                                       - blue thread + extra white fabric in the same grip,
                                       - blue thread partially embedded/hidden inside the white fabric (any part of blue visible OR you can see the blue is going INTO the gripped section).
                                     Only pick 4 when the gripper is EMPTY or contains ONLY white fabric with NO blue visible anywhere in the grip.
  After 5 (LA lift white edge):      Is the blue thread now DANGLING FREELY in front of the RA — hanging loose, separated from the gauze body, clearly grabbable on its own?
                                       yes (blue hangs free) -> 6
                                       no  -> 8 (LA drop+return, retry with 5 or 1)
                                     STRICT: merely "visible" is NOT enough. If the blue is visible but still tangled in / lying flat against the white gauze (not hanging loose), it does NOT pass — the RA would just catch fabric. Pick 8 and retry 5 (re-lift / grab a different edge) until the blue dangles free.
  After 6 (RA grab exposed blue):    Is anything held in the RA gripper (blue thread OR gauze/white fabric — any part of the gauze counts)?
                                       yes -> 7 (LA release white edge). For Strategy B, grabbing ANY part of the lifted gauze is success — proceed with the handoff. Confirm the final verdict after the 7->9 motion clears the LA out of the right_wrist view.
                                       no  -> RA gripper EMPTY (grab missed entirely): pick ra_home (hardcoded RA back; LA STAYS holding the white edge throughout, blue stays exposed). Then on the NEXT prompt, retry 6 (blue is still dangling). Bounded by the wrong-grab counter; after 3 consecutive ra_home retries, switch to the full reset path (7 -> 9 -> 8) and restart from a fresh 5.
  After 7 (LA released white):       (always proceed to 9 next — let LA fully return to rest before evaluating RA contents).
  After 9 (LA back to rest):         Now that LA is out of the way, examine the right_wrist camera. For STRATEGY B, the RA is transferring the WHOLE gauze that the LA had lifted — so grabbing ANY part of it is a success. Decide based on whether the RA gripper is holding anything at all:
                                       RA gripper holds ANYTHING — the blue thread OR the gauze/white fabric (any part of the gauze body) -> 2x (place into slot N — SUCCESS). The whole gauze is in the grip and gets deposited; it does NOT matter whether you can see the blue line specifically. As long as the RA is holding something, place it.
                                       RA gripper EMPTY (nothing held — the grab missed, or the gauze slipped out during the 7->9 motion): pick ra_home (hardcoded RA back; LA is already at rest from the 9). Then on the NEXT prompt, look at the LEFT_WRIST / top camera for the gauze on the green tray and pick 5 to restart Strategy B (fresh white-edge grab). This is the ONLY failure case for skill 6.
  After 2x (RA at slot):             Is the BLUE THREAD still in the RA gripper AT the slot in ANY form (alone, alongside extra fabric, OR partially embedded in white)?
                                       yes -> 3         no -> 4 (gauze slipped; retry gauze)
  After 3 (RA released into slot):   (always proceed to 4 — clean cleanup, gripper already empty)
  After 4 (RA at rest):              Repeat from 1 or 5 for the next gauze. When slots_filled==[1,2,3],
                                     output 'wait' — the operator ends the episode.

SKILL LEGEND (you MUST pick exactly one of these enum values):
{skills_table}

LEGAL ORDERING RULES
  - First action of an episode (and the start of each new gauze) is 1 OR 5, decided SOLELY by the RIGHT_WRIST camera: blue thread clearly visible/grabbable there -> 1 (Strategy A); not clearly visible in right_wrist -> 5 (Strategy B). Do not use the top camera for this choice.
  - 2x is valid when the BLUE THREAD is present in the RA gripper in ANY form:
       (a) blue alone — clear pinch,
       (b) blue + extra gauze fabric alongside in the same grip,
       (c) blue partially EMBEDDED INSIDE the white fabric (the blue is woven through and only partly visible at the grip point — as long as you can identify it's there).
       All three count as a valid grasp; the placing motion deposits the whole gauze. NEVER pick 2x when the gripper is EMPTY or contains ONLY white fabric with NO blue anywhere.
  - 3 is valid ONLY immediately after a 2x where the blue thread is still present at the slot (any form: alone, alongside fabric, or partially embedded — all fine). NEVER pick 3 from a non-slot pose.
  - 4 is the universal RA recovery: pick it whenever RA is in a degenerate state (empty after a grab, holding non-blue, or slipped at slot). The return path drops anything held back over the green tray.
  - 6 is valid in two cases: (a) immediately after a successful 5 where the blue thread is DANGLING FREELY (hanging loose, separated from the gauze — not merely visible-but-tangled), OR (b) immediately after a ra_home that followed a FAILED-EMPTY 6, provided right_wrist still shows the blue thread dangling free (LA was preserved through the hardcoded RA recovery). If the blue is visible but still tangled in the fabric, do NOT pick 6 — pick 8 and re-lift with 5. Retries bounded by wrong-grab counter (max 3 consecutive 6 -> ra_home patterns before forced full-reset).
  - 7 always follows a 6 that ended with ANYTHING in the RA gripper (blue OR gauze/white fabric). The 7 -> 9 sequence is unconditional; the success-vs-failure check happens AT THE AFTER-9 GATE once the right_wrist view is unobstructed.
  - For Strategy B, grabbing ANY part of the gauze is a SUCCESS (the LA already lifted the whole gauze; RA just needs a hold on it to transfer + place). So After 9 has only TWO branches:
       7 -> 9 -> 2x   (SUCCESS — RA holds ANYTHING, blue or gauze fabric: place the whole gauze into the slot)
       7 -> 9 -> ra_home -> 5   (FAILURE — RA gripper EMPTY: nothing held / slipped during 7->9. Hardcoded RA back, then restart Strategy B with a fresh 5.)
  - 8 is the LA-side recovery: pick it when LA is holding the white edge but the situation is unsalvageable — 5 didn't expose blue (no dangling blue after a 5), the 6-retry budget (ra_home loop) is exhausted, or LA has lost grip entirely.
  - Never place into a slot listed in slots_filled. When all 3 slots are filled, pick 'wait'.
  - When the scene is ambiguous, pick 'wait' rather than guessing.

OUTPUT REQUIREMENT
Return ONLY a JSON object matching the schema with fields: subtask_key, rationale (one short sentence, <=200 chars), and visible_blue_confidence (0.0..1.0). Do not output any other text.
"""

    def _build_response_schema(self):
        types = self._types
        return types.Schema(
            type="OBJECT",
            properties={
                "subtask_key": types.Schema(
                    type="STRING",
                    enum=list(SKILL_ENUM),
                    description="The chosen skill identifier.",
                ),
                "rationale": types.Schema(
                    type="STRING",
                    description="One short sentence (<=200 chars) explaining why.",
                ),
                "visible_blue_confidence": types.Schema(
                    type="NUMBER",
                    description="Confidence (0.0..1.0) that blue thread is visible.",
                ),
            },
            required=["subtask_key", "rationale", "visible_blue_confidence"],
        )

    def _encode_image_bytes(self, hwc_uint8) -> bytes:
        np = self._np
        arr = np.asarray(hwc_uint8)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"Expected HWC RGB uint8 image; got shape={arr.shape} dtype={arr.dtype}"
            )
        img = self._Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def _call_once(
        self,
        images: dict,
        slots_filled: list[int],
        timeout_ctx: dict | None = None,
        inspect_ctx: dict | None = None,
    ) -> PlannerDecision:
        types = self._types

        # Lazy-create the chat on the first call of an episode so the model
        # accumulates context (prior images + its own past JSON decisions)
        # across decisions. reset_history() drops this handle at episode end.
        if self._chat is None:
            # Pro models require some thinking budget (budget=0 errors). Empirically
            # thinking_level='high' is the FASTEST on 3.1-pro-preview (~6s) —
            # 'low'/'medium' hit a ~25s floor because the model retries when
            # constrained. Set via env var GEMINI_THINKING_LEVEL if you want
            # to override (low/medium/high), default high.
            import os as _os
            _level = _os.environ.get("GEMINI_THINKING_LEVEL", "high").lower()
            chat_config_kwargs = dict(
                system_instruction=self._system_text,
                response_mime_type="application/json",
                response_schema=self._response_schema,
                temperature=0.2,
            )
            try:
                chat_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=_level,
                )
            except Exception:
                # Older SDKs may not expose thinking_level; degrade gracefully.
                LOGGER.warning("ThinkingConfig(thinking_level=...) unsupported; using default")
            self._chat = self._client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(**chat_config_kwargs),
            )

        state_summary = {
            "slots_filled": slots_filled,
            "last_chosen_subtask": self._last_chosen_key,
        }
        # GEMINI_PROMPT_STYLE=minimal — strip all strategy-specific guidance
        # from runtime state text (streak warning + inspection prompts get a
        # one-line summary instead of the full detailed version).
        _minimal = os.environ.get("GEMINI_PROMPT_STYLE", "detailed").lower() == "minimal"

        state_text = (
            "Current state:\n"
            f"```json\n{json.dumps(state_summary, indent=2)}\n```"
        )
        if self._labeled_history:
            recent = list(self._labeled_history)[-10:]
            state_text += (
                "\n\nRecent operator feedback (most recent last):\n"
                f"```json\n{json.dumps(recent, indent=2)}\n```"
            )
        # Wrong-grab counter context: when the RA has returned home several
        # times in a row because the previous grab caught only the white part
        # (sequence: 1->4 or 6->4 without an intervening success), nudge the
        # planner away from the same approach.
        if self._wrong_grab_count > 0 and _minimal:
            # Minimal mode: just report the streak number; don't tell Gemini
            # what to do about it.
            state_text += (
                f"\n\nWRONG-GRAB STREAK: the RA has returned home "
                f"{self._wrong_grab_count}/{self._WRONG_GRAB_LIMIT} times in a row "
                "after a failed grab (1 or 6 followed by 4 or ra_home)."
            )
        elif self._wrong_grab_count > 0:
            wg_n = self._wrong_grab_count
            wg_lim = self._WRONG_GRAB_LIMIT
            # The streak counts both 1->4 (Strategy A miss) and 6->4 (Strategy
            # B miss). The recommended escape branches on which strategy is
            # currently active — Gemini decides from the labeled history.
            if wg_n >= wg_lim:
                state_text += (
                    f"\n\nWRONG-GRAB STREAK: failed-grab signals (4/ra_home/7 after 1 or 6) "
                    f"have hit {wg_n} in a row — at or over the limit of {wg_lim}. "
                    "RETRY BUDGET EXHAUSTED.\n"
                    "  - If you were using Strategy B (recent picks include '5' or '6'): "
                    "pick '8' (LA release + back) FIRST. This is the cleanest reset — LA "
                    "lets go of the white edge, both arms return to rest, and '8' also "
                    "resets this wrong-grab counter. THEN on the next prompt restart with "
                    "a FRESH '5' at a different edge angle, or '1' if a clean blue thread "
                    "is visible from the green tray.\n"
                    "  - If you were using Strategy A (recent picks were all '1' from the "
                    "green tray): switch to Strategy B — pick '5' (LA grab the white edge "
                    "and lift) so the blue thread is exposed for the right arm.\n"
                    "  - Do NOT retry the same skill from the same approach.\n"
                    "  - Picking '8' (full LA reset) is the recommended path because it "
                    "clears the counter; otherwise the counter persists into the next "
                    "strategy."
                )
            else:
                state_text += (
                    f"\n\nWRONG-GRAB STREAK: the right arm has shown failed-grab signals "
                    f"(4/ra_home/7 after 1 or 6) {wg_n}/{wg_lim} times in a row. You may "
                    "still retry, but only if you can identify a clear reason to expect a "
                    "different result (gauze has shifted, gripper alignment was off, etc.). "
                    "Otherwise switch strategies (Strategy A -> 5; Strategy B -> 8 to fully "
                    "reset, then fresh 5 on a new edge, or 1 if blue is visible)."
                )

        # Inspection context takes precedence over the (legacy) timeout_ctx.
        # When a timeout fires we now freeze BOTH arms and ask Gemini to
        # inspect the grippers BEFORE moving anywhere. The model picks one
        # specific recovery action; we then execute it.
        if inspect_ctx:
            _orig = inspect_ctx.get("subtask")
            _to = inspect_ctx.get("timeout_s")
            _el = inspect_ctx.get("elapsed_s")
            _is_b = bool(inspect_ctx.get("is_strategy_b"))
            _n = int(inspect_ctx.get("consecutive_timeouts", 1))
            _lim = int(inspect_ctx.get("retry_limit", 5))
            if _minimal:
                # Minimal mode: just report the timeout, no per-skill guidance.
                state_text += (
                    f"\n\n=== TIMEOUT ===\n"
                    f"Subtask {_orig!r} exceeded {_to}s (took {_el}s). "
                    f"Timeout #{_n}/{_lim} for this subtask in a row. "
                    "Both arms are currently frozen. Inspect the cameras and "
                    "pick the next subtask from the available enum values.\n"
                    "=== END TIMEOUT ===\n"
                )
            else:
                state_text += (
                    f"\n\n=== TIMEOUT INSPECTION REQUIRED ===\n"
                    f"Subtask {_orig!r} timed out after {_to}s (took {_el}s). "
                    f"This is timeout #{_n}/{_lim} for this subtask in a row. "
                    "BOTH arms are frozen at their current poses. DO NOT move "
                    "without inspecting the grippers first.\n\n"
                    "Examine ALL THREE camera frames before picking.\n\n"
                )
            if not _minimal and _orig == "1":
                state_text += (
                    ">>> SKILL 1 TIMEOUT (RA grab from green tray, Strategy A).\n"
                    "LA is at rest (no white edge held). Pick based on what RA holds:\n"
                    "  - RA EMPTY: pick 'ra_home'. RA returns home via the "
                    "hardcoded smooth linear path (NOT the learned '4'). "
                    "Nothing to drop, so the learned drop-on-return behavior "
                    "is unneeded — the hardcoded path is faster and more "
                    "predictable. On the next prompt, retry with '1' or "
                    "switch to Strategy B with '5'. DO NOT continue '1' from "
                    "the timed-out pose with an empty gripper — that just "
                    "burns another timer in the same bad spot.\n"
                    "  - RA holds the BLUE THREAD pinched: pick '1' to continue. "
                    "Grasp is good, just needs more time to reach the lift-off "
                    "pose where stop_signal fires.\n"
                    "  - RA holds the WRONG part (white fabric only, no blue): "
                    "pick '1' to continue. The motion is near completion and "
                    "the policy will hit stop_signal shortly; on the next "
                    "prompt Gemini will route to '4' (the LEARNED skill — its "
                    "return path passes over the green tray and DROPS the "
                    "wrong grasp on the way home, which is exactly what we "
                    "want here).\n"
                    "DO NOT pick '8' for a skill-1 timeout — LA is not holding "
                    "anything.\n"
                )
            elif not _minimal and _orig == "6":
                state_text += (
                    ">>> SKILL 6 TIMEOUT (RA grab exposed blue, Strategy B).\n"
                    "LA should STILL be holding the white edge of the gauze "
                    "with the blue thread dangling/exposed. KEEP THE LEFT ARM "
                    "HOLDING — do not release it unless LA itself lost grip.\n\n"
                    "Pick based on what BOTH grippers hold:\n"
                    "  - BOTH grippers EMPTY (LA has dropped the white edge, "
                    "blue is no longer exposed — scene is fully reset): pick "
                    "'ra_home'. RA returns home FIRST via the hardcoded "
                    "smooth linear path. On the next prompt, with RA at home "
                    "and LA at the snapshot pose, Gemini will pick '8' (LA "
                    "release + back). This two-step recovery — RA hardcoded "
                    "first, then LA — cleanly resets BOTH arms before "
                    "restarting Strategy B from '5'. DO NOT pick '6' from "
                    "the current pose — there is nothing to grab. DO NOT "
                    "pick '8' first — moving LA back while RA is still at "
                    "the degenerate timeout pose risks collision. DO NOT "
                    "pick '4' (learned) — the learned trajectory does not "
                    "preserve LA, and skill-4's drop-on-return behavior is "
                    "wasted when the gripper is empty.\n"
                    "  - RA EMPTY, LA still holding white: pick 'ra_home'. "
                    "RA returns home via the hardcoded smooth path; LA is "
                    "held at its snapshot pose THROUGHOUT the recovery. On "
                    "the next prompt — RA at home, LA still holding white — "
                    "Gemini picks '8' (LA release + back) to fully reset "
                    "both arms; then restart Strategy B from '5' on a "
                    "subsequent prompt. DO NOT retry '6' from the post-"
                    "recovery pose — if the first attempt missed, the "
                    "gauze/blue alignment likely needs a fresh white-edge "
                    "grab (5) to re-expose blue at a new angle. DO NOT "
                    "pick '6' from the current pose either — RA is in a "
                    "degenerate spot and continuing causes the arms to "
                    "drag. DO NOT pick '4' (learned) — its trajectory "
                    "does not guarantee LA preservation, and there is "
                    "nothing to drop.\n"
                    "  - RA holds ANYTHING (blue OR white) while LA holds: "
                    "pick '6' to continue. The motion will hit stop_signal "
                    "shortly and the next call will route appropriately "
                    "('3' if blue is pinched at lift-off; '4' learned for "
                    "wrong-grab drop). IGNORE the timeout count in this "
                    "case — continue '6' regardless of how many retries "
                    "have happened, because the policy is moments away "
                    "from completing.\n"
                    "  - LA itself lost grip on the white edge (gauze fell, "
                    "blue no longer exposed in any camera): pick '8'. "
                    "Catastrophic — restart Strategy B from skill 5 on the "
                    "next prompt.\n"
                    "  - DO NOT default to '8' just because both grippers "
                    "look occupied — the LA grip is intentional and needs "
                    "to stay through any RA recovery.\n"
                )
            elif not _minimal and _orig == "5":
                state_text += (
                    ">>> SKILL 5 TIMEOUT (LA grab white edge, Strategy B start).\n"
                    "Skill 5 took too long — abort and re-plan. The lift "
                    "should not take this long; if it has, the grab/lift is "
                    "stuck or failed.\n\n"
                    "ACTION: pick '8' (LA release white edge + return to rest). "
                    "This drops whatever the LA has and brings it home, fully "
                    "clearing the left arm.\n\n"
                    "THEN, on the next prompt (LA back at rest, gauze dropped "
                    "on the green tray), decide the strategy using the "
                    "RIGHT_WRIST camera ONLY:\n"
                    "  - If a BLUE THREAD is clearly visible/grabbable in the "
                    "RIGHT_WRIST view -> pick '1' (Strategy A: RA grabs the "
                    "blue directly).\n"
                    "  - If the blue is NOT clearly visible in right_wrist "
                    "(hidden under/inside the white fabric) -> pick '5' "
                    "(Strategy B: re-lift the white edge to re-expose the "
                    "blue, ideally grabbing a different edge). Ignore the top "
                    "camera for this choice.\n"
                    "DO NOT pick 'ra_home' or '4' for a skill-5 timeout — those "
                    "are RA recoveries; the LEFT arm is the one to recover here.\n"
                )
            elif not _minimal:
                state_text += (
                    f"  '{_orig}'  — Extend / continue. Pick this ONLY if the "
                    "RA (or LA, for skill 5) is firmly holding the correct "
                    "part. The same subtask will restart with a fresh timer. "
                    "Use this when the grasp is good but more time is needed.\n\n"
                    "  '4'  — RA learned return (release + return home; drops "
                    "any held grasp into the green tray on the return path). "
                    "Pick this when RA holds a wrong grab that should be "
                    "released into the tray.\n\n"
                    "  'ra_home'  — RA hardcoded smooth return (linear lerp). "
                    "Pick this when EITHER (a) both grippers are empty and a "
                    "predictable home return is preferred, OR (b) LA needs to "
                    "stay put during RA's return (no learned trajectory "
                    "variance).\n\n"
                    "  '8'  — LA release white edge + back. Pick this when the "
                    "LEFT arm is holding the white edge of a gauze and needs to "
                    "drop it.\n"
                )
            if not _minimal:
                state_text += (
                    "\nIf the situation is genuinely ambiguous and you can't tell, "
                    "default to 'ra_home' (safe hardcoded return home, preserves "
                    "LA). Be cautious. Do NOT pick 'wait' or any other skill in "
                    "this inspection — your choice must be one of the above.\n"
                    "=== END TIMEOUT INSPECTION ===\n"
                )
        elif timeout_ctx:
            _tc_sub = timeout_ctx.get("subtask")
            _tc_elapsed = timeout_ctx.get("elapsed_s")
            _tc_to = timeout_ctx.get("timeout_s")
            _tc_count = int(timeout_ctx.get("consecutive_timeouts", 1))
            _tc_limit = int(timeout_ctx.get("retry_limit", 5))
            base = (
                f"\n\nTIMEOUT EVENT: subtask {_tc_sub!r} exceeded "
                f"{_tc_to}s (took {_tc_elapsed}s). "
                "The right arm has been returned to the home position automatically. "
                f"This is timeout #{_tc_count} in a row for this subtask (limit before "
                f"forced strategy change: {_tc_limit}). "
            )
            if _tc_count < _tc_limit:
                base += (
                    "You MAY retry the same subtask if you think a small adjustment "
                    "(re-aim, wait for the scene to settle) is likely to help. "
                    "Look carefully at the cameras — is the gauze position different now? "
                    "Is the gripper aligned? If retrying, briefly note what you'd do "
                    "differently in your rationale."
                )
            else:
                base += (
                    f"You have hit the consecutive-timeout limit of {_tc_limit}. "
                    "DO NOT retry the same subtask. Switch strategy now — e.g. for skill 1 "
                    "failing repeatedly, choose Strategy B (start with 5 to lift the white "
                    "edge and expose blue); for skill 6 failing repeatedly, abort with 8 "
                    "and restart that gauze from scratch."
                )
            state_text += base
        state_text += "\n\nDecide the next subtask and return ONLY the JSON object."

        parts: list[Any] = []
        for label in ("top", "left_wrist", "right_wrist"):
            parts.append(types.Part.from_text(text=f"--- {label.upper()} CAMERA ---"))
            parts.append(types.Part.from_bytes(
                data=self._encode_image_bytes(images[label]),
                mime_type="image/jpeg",
            ))
        parts.append(types.Part.from_text(text=state_text))

        resp = self._chat.send_message(message=parts)

        raw = (resp.text or "").strip()
        if not raw:
            raise RuntimeError("Gemini returned empty response")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Gemini returned non-JSON: {raw[:200]!r}") from e

        chosen_key = data.get("subtask_key", "wait")
        if chosen_key not in SKILL_ENUM:
            LOGGER.warning("Gemini returned unknown key %r; treating as 'wait'", chosen_key)
            chosen_key = "wait"
        rationale = data.get("rationale", "")
        try:
            confidence = float(data.get("visible_blue_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return PlannerDecision(
            chosen_key=chosen_key,
            rationale=rationale,
            visible_blue_confidence=confidence,
            thinking_summary="",
            latency_s=0.0,
            label=None,
            self_check_revised=False,
        )

    def _log_decision(self, d: PlannerDecision, slots_filled: list[int]) -> None:
        if self._log_path is None:
            return
        record = {
            "ts": time.time(),
            "chosen_key": d.chosen_key,
            "label": d.label,
            "rationale": d.rationale,
            "visible_blue_confidence": d.visible_blue_confidence,
            "latency_s": d.latency_s,
            "slots_filled": slots_filled,
            "labeled_history_snapshot": list(self._labeled_history),
            "dry_run": self.dry_run,
            "model": self.model,
        }
        try:
            with self._log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            LOGGER.warning("Failed to write planner JSONL: %s", e)
