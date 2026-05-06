"""
Non-LLM trusted component that maintains credibility scores and enforces permission levels.
(STRICT + ROBUST VERSION)
"""
from __future__ import annotations

from typing import Dict, List, Set
import math

from models import Recommendation, PermissionLevel, RecommendationAction
import config


class PermissionManager:
    def __init__(self):
        self.credibility: Dict[int, float] = {}
        self.clean_turns: Dict[int, int] = {}
        self.agents_with_decrease_this_turn: Set[int] = set()
        self.permanently_removed: Set[int] = set()

    def initialize_agents(self, agent_ids: List[int]) -> None:
        self.permanently_removed.clear()
        for aid in agent_ids:
            self.credibility[aid] = 1.0
            self.clean_turns[aid] = 0

    # ============================
    # HELPERS
    # ============================

    def _get_action_value(self, action):
        if hasattr(action, "value"):
            return action.value
        return action

    def _valid_agent(self, aid: int) -> bool:
        return aid in self.credibility

    def _safe_float(self, x, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            x = float(x)
            if math.isnan(x) or math.isinf(x):
                return default
            return x
        except Exception:
            return default

    # ============================
    # CORE LOGIC
    # ============================

    def process_recommendation(self, rec: Recommendation) -> None:
        target = rec.target_agent_id

        if not self._valid_agent(target):
            return
        if target in self.permanently_removed:
            return

        action = self._get_action_value(rec.action)

        confidence = self._safe_float(rec.confidence, 0.0)
        confidence = max(0.0, min(1.0, confidence))
        if confidence < config.DETECTOR_ACTION_MIN_CONFIDENCE:
            return

        # magnitude tied to confidence (bounded)
        raw_magnitude = rec.magnitude if rec.magnitude is not None else 0.1
        magnitude = self._safe_float(raw_magnitude, 0.1)
        magnitude = max(0.0, min(0.3, magnitude))
        magnitude = min(magnitude, confidence)

        current = self.credibility[target]

        # ============================
        # DECREASE
        # ============================
        if action == RecommendationAction.DECREASE.value:
            if current <= 0.0:
                return

            new_val = max(0.0, current - magnitude)

            if new_val != current:
                self.credibility[target] = new_val
                self.agents_with_decrease_this_turn.add(target)

        # ============================
        # QUARANTINE
        # ============================
        elif action == RecommendationAction.QUARANTINE.value:
            threshold = config.CREDIBILITY_QUARANTINE + 0.1
            self.agents_with_decrease_this_turn.add(target)

            if current <= threshold:
                return  # already quarantined; keep recovery suppressed this turn

            self.credibility[target] = threshold

        # ============================
        # REMOVE
        # ============================
        elif action == RecommendationAction.REMOVE.value:
            self.agents_with_decrease_this_turn.add(target)
            if current == 0.0:
                return  # already removed; keep recovery suppressed this turn

            self.credibility[target] = 0.0
            self.clean_turns[target] = 0
            self.permanently_removed.add(target)

    # ============================
    # TURN UPDATE
    # ============================

    def end_turn(self) -> None:
        for aid in list(self.credibility.keys()):
            if aid in self.permanently_removed:
                self.clean_turns[aid] = 0
                continue
            if aid in self.agents_with_decrease_this_turn:
                self.clean_turns[aid] = 0
                continue

            self.clean_turns[aid] += 1

            if self.clean_turns[aid] >= config.CLEAN_TURNS_FOR_RECOVERY:
                self.credibility[aid] = min(
                    1.0,
                    self.credibility[aid] + config.RECOVERY_INCREMENT
                )
                self.clean_turns[aid] = 0

        self.agents_with_decrease_this_turn.clear()

    # ============================
    # PERMISSION
    # ============================

    def get_permission_level(self, agent_id: int, is_language_only: bool = False) -> PermissionLevel:
        c = self.credibility.get(agent_id, 1.0)

        if c >= config.CREDIBILITY_FULL:
            return PermissionLevel.FULL_ACCESS

        if is_language_only and c >= config.CREDIBILITY_RESTRICTED:
            # In QA mode we do not have a meaningful "restricted but still
            # partially capable" execution state. Treat the restricted band as
            # the re-entry threshold so quarantined agents can actually recover
            # participation within the experiment horizon.
            return PermissionLevel.FULL_ACCESS

        if c >= config.CREDIBILITY_RESTRICTED:
            return PermissionLevel.RESTRICTED

        if c >= config.CREDIBILITY_QUARANTINE:
            return PermissionLevel.QUARANTINE

        return PermissionLevel.REMOVED
