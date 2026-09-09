"""Shared skill vocabulary and structured planner result."""
from dataclasses import dataclass

# Skill enum keys. "1"-"9" map directly into SUBTASK_LABELS; "2a/2b/2c" expand
# the slot-templated key 2 by resolving {N} → 1/2/3. "wait" is a no-op
# (planner stays silent so the operator can take over).
SKILL_ENUM = ["1", "2a", "2b", "2c", "3", "4", "5", "6", "7", "8", "9", "ra_home", "wait"]

SKILL_DESCRIPTIONS: dict[str, str] = {
    "1":  "RA grab the nearest blue thread directly from the green tray. Use ONLY when the blue thread is clearly visible from the right wrist camera.",
    "2a": "RA lower and place the gauze into slot 1 of the counting tray. Precondition: RA gripper must include the BLUE THREAD in its grip in ANY form: (a) blue alone, (b) blue + extra gauze fabric in the same grip, OR (c) blue PARTIALLY EMBEDDED inside the white fabric (the blue is woven through and only partially visible — still counts as held). All three count as a valid grasp; the placing motion deposits the whole gauze regardless. Only pick 4 instead if the gripper is empty OR is holding ONLY white fabric/edge with NO blue thread visible anywhere in the grip.",
    "2b": "RA lower and place the gauze into slot 2 of the counting tray. Precondition: RA gripper must include the BLUE THREAD in its grip in ANY form: (a) blue alone, (b) blue + extra gauze fabric in the same grip, OR (c) blue PARTIALLY EMBEDDED inside the white fabric (the blue is woven through and only partially visible — still counts as held). All three count as a valid grasp; the placing motion deposits the whole gauze regardless. Only pick 4 instead if the gripper is empty OR is holding ONLY white fabric/edge with NO blue thread visible anywhere in the grip.",
    "2c": "RA lower and place the gauze into slot 3 of the counting tray. Precondition: RA gripper must include the BLUE THREAD in its grip in ANY form: (a) blue alone, (b) blue + extra gauze fabric in the same grip, OR (c) blue PARTIALLY EMBEDDED inside the white fabric (the blue is woven through and only partially visible — still counts as held). All three count as a valid grasp; the placing motion deposits the whole gauze regardless. Only pick 4 instead if the gripper is empty OR is holding ONLY white fabric/edge with NO blue thread visible anywhere in the grip.",
    "3":  "RA open the right gripper to drop the gauze INTO the counting-tray slot. This is the SUCCESS-path release, used at the end of a clean 2a/2b/2c. Precondition: right_wrist camera confirms the BLUE THREAD is still in the RA grip AT the slot in ANY form — alone, alongside extra gauze fabric, OR partially embedded inside the white fabric (all three count as valid; the whole gauze gets deposited together). Pick 4 next (clean return to rest). NEVER pick 3 if the gripper is empty or holding ONLY white fabric with NO blue anywhere — pick 4 instead so the arm retreats and any wrong grasp is dropped back to the green tray.",
    "4":  "RA return to rest position. Dual purpose: (a) STANDARD CLEANUP after a successful 3 release — gripper is empty, simply travel home; (b) RECOVERY / DROP-BACK when the RA gripper is empty after a failed 1/6, OR is holding any non-blue part of the gauze, OR the gauze slipped during 2x — the arm's return path passes over the green tray and drops anything held back into the source on the way out. Use 4 anytime RA is in a degenerate state (after a missed grab, wrong-part grab, or slipped place). Restart the gauze afterwards with 1 or 5.",
    "5":  "LA grab the WHITE edge of a gauze and lift it. Use when the blue thread is NOT visible — lifting the white edge should expose a FREELY DANGLING blue thread for the right arm to grab next. Success of 5 requires the blue to hang loose/separated from the gauze body (not merely visible while still tangled in the fabric); if it's still tangled, re-lift (8 then 5).",
    "6":  "RA grab the EXPOSED blue thread. Precondition: skill 5 must have just completed (left arm is currently holding a gauze up by its white edge).",
    "7":  "LA release the white edge (open left gripper). Pick this WHENEVER skill 6 ended with ANYTHING in the RA gripper — for Strategy B, the RA grabbing ANY part of the lifted gauze (blue OR white fabric) is a success. Followed unconditionally by 9 (LA back to rest), then the after-9 verdict has only two outcomes: 2x if the RA is still holding anything (success — place the whole gauze), or ra_home if the RA gripper became empty during the 7->9 motion (the only failure case; restart Strategy B from a fresh 5).",
    "8":  "LA release the white edge AND return to rest (drop & abort). Recovery skill — pick this when EITHER (a) skill 5 just completed but no blue thread is dangling/visible from the RIGHT_WRIST camera (the lift didn't expose blue, so skill 6 would fail), OR (b) skill 6 was attempted and the RIGHT arm gripper did not capture the blue thread (visible in right_wrist after the attempt). The gauze gets dropped back onto the green tray and you start over with that gauze.",
    "9":  "LA return to rest position.",
    "ra_home": "HARDCODED smooth RA return to home pose (linear lerp, not policy-driven). LA is held still at its current pose throughout. NO release functionality — anything in the RA gripper stays there. Use ONLY when EITHER (a) RA is empty and you need a clean predictable home return (no learned trajectory variance), OR (b) LA is holding the gauze white edge and MUST be preserved during RA's return (Strategy B recovery). DO NOT use this when RA holds a wrong grab that should be dropped into the green tray — use '4' (learned) for that, since '4' drops the grasp on the return path.",
    "wait": "Do nothing this round. Use when the scene is unclear, a gauze is mid-motion and you need more information, or when you would otherwise pick an unsafe skill.",
}

# Concise one-line skill descriptions for the "minimal prompt" Gemini mode
# (GEMINI_PROMPT_STYLE=minimal). Used to test how the planner performs with
# only goal + skill names + minimal semantics, no detailed strategy logic.
SKILL_DESCRIPTIONS_MINIMAL: dict[str, str] = {
    "1":  "Right arm grabs the blue thread directly from the green tray.",
    "2a": "Right arm places the held gauze into slot 1 of the counting tray.",
    "2b": "Right arm places the held gauze into slot 2 of the counting tray.",
    "2c": "Right arm places the held gauze into slot 3 of the counting tray.",
    "3":  "Right arm opens the gripper to release the gauze into the slot.",
    "4":  "Right arm DROP AND RETURN (keyboard key 4): the return path drops any held grasp back over the green tray, then the arm returns to rest.",
    "5":  "Left arm grabs the white edge of a gauze and lifts to expose the blue thread.",
    "6":  "Right arm grabs the exposed blue thread (after a 5).",
    "7":  "Left arm opens the gripper to release the white edge.",
    "8":  "Left arm releases the white edge AND returns to rest in one motion (combined recovery).",
    "9":  "Left arm returns to rest position.",
    "ra_home": "Right arm HARDCODED RETURN: go back to the default (home) position. Does NOT drop anything (anything held stays in the gripper); left arm held still throughout — preserves any LA grasp.",
    "wait": "Do nothing this round.",
}


# Slot-suffix keys → numeric slot for {N} resolution.
SLOT_NUMBERS: dict[str, int] = {"2a": 1, "2b": 2, "2c": 3}



@dataclass
class PlannerDecision:
    chosen_key: str               # one of SKILL_ENUM
    rationale: str
    visible_blue_confidence: float  # 0..1, set by pass 1
    thinking_summary: str         # concatenated summarized-thinking text
    latency_s: float
    label: str | None             # resolved SUBTASK_LABELS string (or None for "wait" / dry_run)
    self_check_revised: bool      # True if pass-2 changed pass-1's choice
