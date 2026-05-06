"""
Peer assignment for the decentralized Knowledge-QA worker pool.
"""
from __future__ import annotations

import random
from typing import List, Optional

import config
from models import Agent


def assign_roles(domain: str, num_workers: int = None, rng: Optional[random.Random] = None) -> List[Agent]:
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if domain not in {"knowledge_qa", "code_synthesis"}:
        raise ValueError("Unsupported domain")

    agents: List[Agent] = []
    for i in range(num_workers):
        role = "developer" if domain == "code_synthesis" else None
        agents.append(Agent(
            id=i,
            role=role,
            is_malicious=False,
        ))
    return agents


def malicious_count_from_label(m: int | str, total_agents: int) -> int:
    if isinstance(m, str) and m.strip().lower() == "all":
        return total_agents
    count = int(m)
    if count < 0 or count > total_agents:
        raise ValueError(f"m must be between 0 and {total_agents}, or 'all'")
    return count


def mark_malicious(
    agents: List[Agent],
    m: int | str,
    rng: Optional[random.Random] = None,
) -> List[Agent]:
    count = malicious_count_from_label(m, len(agents))
    for agent in agents:
        agent.is_malicious = False
    if count == 0:
        return agents
    if count == len(agents):
        for agent in agents:
            agent.is_malicious = True
        return agents

    chooser = rng or random
    indices = sorted(chooser.sample(list(range(len(agents))), count))
    for idx in indices:
        agents[idx].is_malicious = True
    return agents
