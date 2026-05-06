import os
import random
import time
from threading import Lock

lock = Lock()
last_call_by_key: dict[str, float] = {}


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _rate_limit_per_sec() -> float:
    # The full HotpotQA matrix is call-heavy; default to modest Azure parallelism.
    # Override RATE_LIMIT_PER_SEC if a deployment needs stricter throttling.
    return max(_get_env_float("RATE_LIMIT_PER_SEC", 2.0), 0.01)


def _max_retries() -> int:
    return max(_get_env_int("RATE_LIMIT_MAX_RETRIES", 8), 1)


def _base_backoff() -> float:
    return max(_get_env_float("RATE_LIMIT_BACKOFF_BASE", 2.0), 0.1)


def _max_backoff() -> float:
    return max(_get_env_float("RATE_LIMIT_BACKOFF_MAX", 60.0), 1.0)


def _reserve_request_slot(limiter_key: str) -> None:
    wait = 0.0
    with lock:
        min_interval = 1.0 / _rate_limit_per_sec()
        now = time.monotonic()
        last_call = last_call_by_key.get(limiter_key, 0.0)
        reserved_time = max(now, last_call + min_interval)
        last_call_by_key[limiter_key] = reserved_time
        wait = max(0.0, reserved_time - now)
    if wait > 0:
        time.sleep(wait)


def _is_retryable_throttle(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in (
        "429",
        "too many requests",
        "too_many_requests",
        "rate limit",
        "ratelimit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    ))


def rate_limited_call(func, *args, limiter_key: str | None = None, **kwargs):
    max_retries = _max_retries()
    bucket = str(limiter_key or "__global__")

    for attempt in range(max_retries):
        _reserve_request_slot(bucket)
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries - 1 or not _is_retryable_throttle(exc):
                raise

            sleep_for = min(_max_backoff(), _base_backoff() * (2 ** attempt))
            sleep_for += random.uniform(0.0, min(1.0, sleep_for * 0.25))
            time.sleep(sleep_for)
