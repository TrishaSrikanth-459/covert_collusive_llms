"""
Core data models for the collusive covert-attack Knowledge-QA study.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class AttackType(str, Enum):
    NONE = "none"
    SUBOPTIMAL_FIXATION = "suboptimal_fixation"
    REFRAMING_MISALIGNMENT = "reframing_misalignment"
    FAKE_INJECTION = "fake_injection"
    # Legacy-only: retained so older logs/results remain parseable, but the
    # current runner no longer schedules or supports this attack.
    EXECUTION_DELAY = "execution_delay"

    # Backward-compatible names from earlier pilots. They are not emitted by the
    # new runner, but keeping them prevents old logs/scripts from crashing.
    LINGUISTIC_COLLUSION = "suboptimal_fixation"
    MULTIMODAL_HIDING = "reframing_misalignment"
    FALSE_CONSENSUS_INJECTION = "fake_injection"

class PermissionLevel(str, Enum):
    FULL_ACCESS = "full_access"
    RESTRICTED = "restricted"
    QUARANTINE = "quarantine"
    REMOVED = "removed"


class RecommendationAction(str, Enum):
    DECREASE = "decrease_credibility"
    QUARANTINE = "quarantine"
    REMOVE = "remove"


class Agent(BaseModel):
    id: int
    role: Optional[str] = None
    is_malicious: bool = False
    hidden_prompt: Optional[str] = None

class Message(BaseModel):
    turn: int
    sender_id: int
    content: str
    is_private: bool = False
    recipient_id: Optional[int] = None
    timestamp: float = None

    def __init__(self, **data):
        if data.get("timestamp") is None:
            data["timestamp"] = time.time()
        super().__init__(**data)


class Action(BaseModel):
    turn: int
    agent_id: int
    action_type: str
    content: str
    timestamp: float = None

    def __init__(self, **data):
        if data.get("timestamp") is None:
            data["timestamp"] = time.time()
        super().__init__(**data)


class Recommendation(BaseModel):
    target_agent_id: int
    action: str
    magnitude: Optional[float] = None
    confidence: float
    detector_ids: List[int]
    evidence: str
    turn: int
    timestamp: float = None

    def __init__(self, **data):
        if data.get("timestamp") is None:
            data["timestamp"] = time.time()
        super().__init__(**data)


class PermissionChange(BaseModel):
    timestamp: float
    agent_id: int
    old_credibility: float
    new_credibility: float
    reason: str
    recommending_detectors: Optional[List[int]] = None
