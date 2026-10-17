#!/usr/bin/env python3
"""Resumable model download for hf-download.sh (Python 3.10+, stdlib + huggingface_hub).

Calls `snapshot_download` from a SINGLE long-lived process so a transient
network failure does not tear the transfer down and start from zero. Retrying
inside one process keeps the Xet CAS session and chunk cache warm, so a later
attempt reuses previously downloaded chunks for partially reconstructed files.

The retry policy is delegated here (not shell): retries are bounded or
unlimited, with exponential backoff doubling from an initial delay and capped
at a maximum.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable


def compute_delay(delay: int, max_delay: int) -> int:
    doubled = delay * 2
    return max_delay if doubled > max_delay else doubled


def retry_download(
    repo_id: str,
    *,
    cache_dir: str,
    retries: int,
    unlimited: bool,
    retry_delay: int,
    retry_max_delay: int,
    snapshot_download: Callable[..., object],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Download `repo_id` with bounded/unlimited retry and exponential backoff.

    `snapshot_download` is injected so tests can stub it without network.
    Raises RuntimeError when the retry budget is exhausted (never if `unlimited`).
    """
    attempt = 1
    delay = retry_delay
    while True:
        try:
            snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
            return
        except Exception:
            if not unlimited and attempt > retries:
                raise RuntimeError(
                    f"Failed to download model '{repo_id}' after {attempt} attempts."
                )
            if unlimited:
                print(
                    f"Download failed (attempt {attempt}); retrying forever in {delay}s...",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Download failed (attempt {attempt}); retrying ({attempt}/{retries}) in {delay}s...",
                    file=sys.stderr,
                )
            sleep(delay)
            delay = compute_delay(delay, retry_max_delay)
            attempt += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable Hugging Face model download.")
    parser.add_argument("repo_id", help="HuggingFace repo id (e.g. org/model).")
    parser.add_argument("--cache-dir", required=True, help="HF hub cache directory.")
    parser.add_argument("--retries", default="3",
                        help="Retries after a failed download; 'unlimited' (or 'forever'/'infinite') retries forever.")
    parser.add_argument("--retry-delay", type=int, default=5,
                        help="Initial delay before the first retry (seconds).")
    parser.add_argument("--retry-max-delay", type=int, default=1800,
                        help="Backoff cap (seconds, default 30min).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.retries not in ("unlimited", "forever", "infinite") and not args.retries.isdigit():
        print(f"--retries must be a non-negative integer or 'unlimited' (got: {args.retries!r})",
              file=sys.stderr)
        return 2
    if args.retry_delay < 0 or args.retry_max_delay < 0:
        print("--retry-delay and --retry-max-delay must be non-negative integers.", file=sys.stderr)
        return 2

    from huggingface_hub import snapshot_download as real_snapshot_download

    retries: int = 0 if args.retries in ("unlimited", "forever", "infinite") else int(args.retries)
    try:
        retry_download(
            args.repo_id,
            cache_dir=args.cache_dir,
            retries=retries,
            unlimited=args.retries in ("unlimited", "forever", "infinite"),
            retry_delay=args.retry_delay,
            retry_max_delay=args.retry_max_delay,
            snapshot_download=real_snapshot_download,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
