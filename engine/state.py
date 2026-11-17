"""
Redis-backed crawl state for the DarkTrace Web Crawler.

Week 6 compatibility:
    - Bulk scheduler snapshot
    - Due checking with failure backoff
    - In-progress/stale-job detection
    - Redis scheduler locking
    - ETag / Last-Modified incremental crawling
    - Content hash tracking
    - Success / retry / dead-letter counters
    - Source health state

Week 7 Stage 3:
    - Crawl run ID
    - Interrupted crawl detection
    - Recovery state
    - Recovery counters
    - Persistent crawl-start information

Week 8:
    - Durable pending-URL frontier and completed-page checkpoints

One Redis hash per source:

    crawlstate:<source_name>

The source name remains the logical identity of the crawl.
Recovery does NOT create a new source.
"""

import json
import math
import os
import time
import uuid

import redis


REDIS_URL = os.environ.get(
    "CRAWL_STATE_REDIS_URL",
    "redis://localhost:6379/2",
)

MAX_BACKOFF_FACTOR = 8


class CrawlState:

    def __init__(self, redis_url: str = REDIS_URL):
        self.r = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )

    # ==================================================================
    # Redis keys
    # ==================================================================

    def _key(self, source_name: str) -> str:
        return f"crawlstate:{source_name}"

    def _lock_key(self, source_name: str) -> str:
        return f"crawl-lock:{source_name}"

    # ==================================================================
    # Week 6 scheduler bulk state
    # ==================================================================

    def snapshot(self, source_names: list[str]) -> dict:
        """
        Fetch all source states in one Redis pipeline.
        Required by Week 6 scheduler.
        """
        if not source_names:
            return {}

        pipe = self.r.pipeline(transaction=False)

        for name in source_names:
            pipe.hgetall(self._key(name))

        return dict(
            zip(
                source_names,
                pipe.execute(),
            )
        )

    @staticmethod
    def in_progress_from(
        state: dict,
        stale_after: float,
    ) -> bool:
        """
        Return True only when the source is genuinely running.

        Missing or malformed timestamps cannot prove that a crawl is
        still active, so they are treated as stale.
        """
        return (
            state.get("in_progress") == "1"
            and not CrawlState.stale_in_progress_from(
                state,
                stale_after,
            )
        )

    @staticmethod
    def stale_in_progress_from(
        state: dict,
        stale_after: float,
        now: float | None = None,
    ) -> bool:
        if state.get("in_progress") != "1":
            return False

        try:
            started = float(state["started_ts"])
        except (KeyError, TypeError, ValueError):
            return True

        if not math.isfinite(started):
            return True

        current_time = time.time() if now is None else now
        return current_time - started >= stale_after

    @staticmethod
    def due_from(
        state: dict,
        interval_minutes: int,
    ) -> bool:
        """
        Week 6 due calculation with exponential failure backoff.
        """
        last = state.get("last_crawled_ts")

        if last is None:
            return True

        try:
            failures = int(
                state.get(
                    "consecutive_failures",
                    0,
                ) or 0
            )
        except (TypeError, ValueError):
            failures = 0

        factor = min(
            2 ** failures,
            MAX_BACKOFF_FACTOR,
        )

        return (
            time.time() - float(last)
        ) >= interval_minutes * 60 * factor

    # ==================================================================
    # Single-source compatibility helpers
    # ==================================================================

    def is_in_progress(
        self,
        source_name: str,
        stale_after: float = 7200,
    ) -> bool:
        return self.in_progress_from(
            self.get_health(source_name),
            stale_after,
        )

    def is_due(
        self,
        source_name: str,
        interval_minutes: int,
    ) -> bool:
        return self.due_from(
            self.get_health(source_name),
            interval_minutes,
        )

    # ==================================================================
    # Week 6 scheduler locking
    # ==================================================================

    def acquire_lock(
        self,
        source_name: str,
        ttl: int = 3600,
    ) -> bool:
        """
        Atomically claim a source.
        """
        return bool(
            self.r.set(
                self._lock_key(source_name),
                "1",
                nx=True,
                ex=ttl,
            )
        )

    def release_lock(
        self,
        source_name: str,
    ) -> None:
        self.r.delete(
            self._lock_key(source_name)
        )

    # ==================================================================
    # Crawl lifecycle
    # ==================================================================

    def mark_started(
        self,
        source_name: str,
        recovered: bool = False,
    ) -> str:
        """
        Start a crawl.

        Recovery is explicit so a normal Celery retry does not increase
        the crash-recovery counter.

        Returns the new crawl_run_id.
        """
        key = self._key(source_name)

        previous_recovery_count = self.r.hget(
            key,
            "recovery_count",
        )

        try:
            recovery_count = int(
                previous_recovery_count or 0
            )
        except (TypeError, ValueError):
            recovery_count = 0

        now = time.time()
        run_id = str(uuid.uuid4())

        mapping = {
            "in_progress": "1",
            "started_ts": now,
            "last_started_ts": now,
            "last_status": "in_progress",
            "crawl_run_id": run_id,
        }

        if recovered:
            recovery_count += 1

            mapping.update(
                {
                    "recovery_pending": "1",
                    "recovery_count": recovery_count,
                    "last_recovery_ts": now,
                }
            )
        elif self.r.hget(key, "recovery_pending") is None:
            mapping["recovery_pending"] = "0"

        if previous_recovery_count is None:
            mapping["recovery_count"] = recovery_count

        self.r.hset(
            key,
            mapping=mapping,
        )

        return run_id

    def mark_success(
        self,
        source_name: str,
        record: dict | None = None,
    ) -> None:
        """
        Record successful crawl.

        Week 6:
            Stores ETag, Last-Modified and content hash.

        Week 7:
            Clears recovery-pending state and resets failures.
        """
        now = time.time()
        key = self._key(source_name)

        pipe = self.r.pipeline()

        pipe.hset(
            key,
            mapping={
                "last_crawled_ts": now,
                "last_success_ts": now,
                "last_status": "success",
                "consecutive_failures": 0,
                "recovery_pending": "0",
            },
        )

        pipe.hdel(
            key,
            "last_error",
        )

        pipe.hincrby(
            key,
            "total_success",
            1,
        )

        if record:
            validators = {
                "etag": record.get("etag"),
                "last_modified": record.get(
                    "last_modified"
                ),
                "last_content_hash": record.get(
                    "content_hash"
                ),
            }

            for field, value in validators.items():

                if value:
                    pipe.hset(
                        key,
                        field,
                        value,
                    )
                else:
                    pipe.hdel(
                        key,
                        field,
                    )

        pipe.execute()

        self.clear_recovery(source_name)

    def mark_retry(
        self,
        source_name: str,
        reason: str,
    ) -> None:
        """
        Mark temporary whole-job failure.

        The source remains in progress because Celery will retry it.
        """
        self.r.hset(
            self._key(source_name),
            mapping={
                "last_status": "retrying",
                "last_error": reason[:500],
            },
        )

    def mark_dead_letter(
        self,
        source_name: str,
        reason: str,
    ) -> None:
        """
        Permanently failed crawl after retries are exhausted.
        """
        now = time.time()
        key = self._key(source_name)

        pipe = self.r.pipeline()

        pipe.hset(
            key,
            mapping={
                "last_crawled_ts": now,
                "last_status": "dead_letter",
                "last_error": reason[:500],
                "recovery_pending": "0",
            },
        )

        pipe.hincrby(
            key,
            "total_failed",
            1,
        )

        pipe.hincrby(
            key,
            "consecutive_failures",
            1,
        )

        pipe.execute()

    def mark_finished(
        self,
        source_name: str,
    ) -> None:
        """
        End the current worker execution.

        Lock release remains the caller's responsibility.
        """
        self.r.hset(
            self._key(source_name),
            "in_progress",
            "0",
        )

    # ==================================================================
    # Incremental crawling
    # ==================================================================

    def get_validators(
        self,
        source_name: str,
    ) -> dict:
        """
        Return validators from the last successful crawl.

        Required by Week 6 conditional-request behavior.
        """
        values = self.r.hmget(
            self._key(source_name),
            "etag",
            "last_modified",
            "last_content_hash",
        )

        etag, last_modified, content_hash = values

        if not content_hash:
            return {}

        if not (
            etag
            or last_modified
        ):
            return {}

        return {
            "etag": etag,
            "last_modified": last_modified,
            "prev_hash": content_hash,
        }

    # ==================================================================
    # Week 7 recovery
    # ==================================================================

    def needs_recovery(
        self,
        source_name: str,
    ) -> bool:
        """
        True when the source has a crawl that was left in progress.
        """
        return (
            self.r.hget(
                self._key(source_name),
                "in_progress",
            )
            == "1"
        )

    def mark_recovery_pending(
        self,
        source_name: str,
        reason: str = "crawl interrupted",
    ) -> None:
        now = time.time()

        self.r.hset(
            self._key(source_name),
            mapping={
                "in_progress": "1",
                "last_status": "recovery_pending",
                "recovery_pending": "1",
                "last_recovery_ts": now,
                "last_error": reason[:500],
            },
        )

    def mark_checkpoint(
        self,
        source_name: str,
        checkpoint: str | None = None,
    ) -> None:
        mapping: dict[str, str | float] = {
            "last_checkpoint_ts": time.time(),
        }

        if checkpoint is not None:
            mapping["last_checkpoint"] = str(
                checkpoint
            )[:1000]

        self.r.hset(
            self._key(source_name),
            mapping=mapping,
        )

    def get_crawl_checkpoint(
        self,
        source_name: str,
    ) -> dict | None:
        raw = self.r.hget(
            self._key(source_name),
            "last_checkpoint",
        )

        if raw is None:
            return None

        if not isinstance(raw, str):
            raise ValueError(
                f"Checkpoint for source '{source_name}' is not text"
            )

        checkpoint = json.loads(raw)

        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("version") != 1
            or not isinstance(checkpoint.get("seed_url"), str)
            or not isinstance(checkpoint.get("pending_urls"), list)
            or not isinstance(checkpoint.get("visited_urls"), list)
            or not isinstance(checkpoint.get("processed_count"), int)
        ):
            raise ValueError(
                f"Invalid crawl checkpoint for source '{source_name}'"
            )

        if any(
            not isinstance(url, str)
            for url in (
                checkpoint["pending_urls"]
                + checkpoint["visited_urls"]
            )
        ):
            raise ValueError(
                f"Invalid URLs in crawl checkpoint for source '{source_name}'"
            )

        seed_record = checkpoint.get("seed_record")
        if seed_record is not None and not isinstance(seed_record, dict):
            raise ValueError(
                f"Invalid seed result in crawl checkpoint for source '{source_name}'"
            )

        return checkpoint

    def save_crawl_checkpoint(
        self,
        source_name: str,
        pending_urls: list[str],
        visited_urls: list[str],
        processed_count: int,
        seed_url: str,
    ) -> None:
        key = self._key(source_name)
        previous = self.get_crawl_checkpoint(source_name)
        checkpoint = {
            "version": 1,
            "seed_url": seed_url,
            "pending_urls": list(dict.fromkeys(pending_urls)),
            "visited_urls": list(dict.fromkeys(visited_urls)),
            "processed_count": max(
                0,
                int(processed_count),
                previous.get("processed_count", 0)
                if previous
                else 0,
            ),
            "updated_ts": time.time(),
        }

        if (
            previous
            and previous.get("seed_url") == seed_url
            and previous.get("seed_record")
        ):
            checkpoint["seed_record"] = previous["seed_record"]

        pipe = self.r.pipeline()
        pipe.hset(
            key,
            mapping={
                "last_checkpoint": json.dumps(
                    checkpoint,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "last_checkpoint_ts": checkpoint["updated_ts"],
            },
        )
        pipe.execute()

    def complete_checkpoint_url(
        self,
        source_name: str,
        url: str,
        seed_record: dict | None = None,
    ) -> int:
        checkpoint = self.get_crawl_checkpoint(source_name)

        if checkpoint is None:
            raise ValueError(
                f"Cannot complete '{url}': crawl checkpoint is missing "
                f"for source '{source_name}'"
            )

        pending_urls = checkpoint["pending_urls"]

        if url in pending_urls:
            checkpoint["pending_urls"] = [
                pending_url
                for pending_url in pending_urls
                if pending_url != url
            ]
            checkpoint["processed_count"] += 1

        if seed_record is not None:
            checkpoint["seed_record"] = seed_record

        checkpoint["updated_ts"] = time.time()
        self.r.hset(
            self._key(source_name),
            mapping={
                "last_checkpoint": json.dumps(
                    checkpoint,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "last_checkpoint_ts": checkpoint["updated_ts"],
            },
        )
        return checkpoint["processed_count"]

    def clear_crawl_checkpoint(
        self,
        source_name: str,
    ) -> None:
        self.r.hdel(
            self._key(source_name),
            "last_checkpoint",
            "last_checkpoint_ts",
        )

    def clear_recovery(
        self,
        source_name: str,
    ) -> None:
        key = self._key(source_name)

        self.r.hset(
            key,
            "recovery_pending",
            "0",
        )

        self.r.hdel(
            key,
            "last_checkpoint",
            "last_checkpoint_ts",
            "last_recovery_ts",
        )

    def get_recovery_state(
        self,
        source_name: str,
    ) -> dict:
        data = self.get_health(
            source_name
        )

        return {
            "source_name": source_name,
            "in_progress": data.get(
                "in_progress",
                "0",
            ),
            "last_status": data.get(
                "last_status"
            ),
            "crawl_run_id": data.get(
                "crawl_run_id"
            ),
            "last_started_ts": data.get(
                "last_started_ts"
            ),
            "last_checkpoint_ts": data.get(
                "last_checkpoint_ts"
            ),
            "last_checkpoint": data.get(
                "last_checkpoint"
            ),
            "recovery_pending": data.get(
                "recovery_pending",
                "0",
            ),
            "recovery_count": data.get(
                "recovery_count",
                "0",
            ),
            "last_recovery_ts": data.get(
                "last_recovery_ts"
            ),
            "last_error": data.get(
                "last_error"
            ),
        }

    def get_crawl_run_id(
        self,
        source_name: str,
    ) -> str | None:
        return self.r.hget(
            self._key(source_name),
            "crawl_run_id",
        )

    # ==================================================================
    # Health / inspection
    # ==================================================================

    def get_health(
        self,
        source_name: str,
    ) -> dict:
        return self.r.hgetall(
            self._key(source_name)
        )

    def get_health_status(
        self,
        source_name: str,
    ) -> str:
        """
        Simple compatibility helper for health tests.
        """
        data = self.get_health(
            source_name
        )

        status = data.get(
            "last_status"
        )

        if status == "success":
            return "recovered"

        if status == "dead_letter":
            return "unavailable"

        if status in {
            "retrying",
            "failed",
        }:
            return "degraded"

        return status or "unknown"

    def all_states(
        self,
        source_names: list[str],
    ) -> dict:
        return self.snapshot(
            source_names
        )