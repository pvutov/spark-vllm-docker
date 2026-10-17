#!/usr/bin/env python3
"""Unit tests for hf-resume-download.py retry/backoff logic (no network)."""

from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
hd_source = (ROOT / "hf-resume-download.py").read_text()
hd = types.ModuleType("hf_resume_download")
exec(compile(hd_source, "hf-resume-download.py", "exec"), hd.__dict__)


def make_snapshot_download(fail_count: int):
    """Return a snapshot_download stub that fails `fail_count` times, then succeeds."""
    state = {"calls": 0}

    def stub(repo_id=None, cache_dir=None, **kwargs):
        state["calls"] += 1
        if state["calls"] <= fail_count:
            raise ConnectionError("boom")
        return str(cache_dir)

    return stub, state


class RetryDownloadTests(unittest.TestCase):
    def run_with(self, snapshot_download, *, retries=3, unlimited=False,
                 retry_delay=5, retry_max_delay=1800):
        sleeps = []
        hd.retry_download(
            "org/model",
            cache_dir="/cache",
            retries=retries,
            unlimited=unlimited,
            retry_delay=retry_delay,
            retry_max_delay=retry_max_delay,
            snapshot_download=snapshot_download,
            sleep=lambda s: sleeps.append(s),
        )
        return sleeps

    def test_succeeds_after_transient_failures(self):
        stub, state = make_snapshot_download(2)
        sleeps = self.run_with(stub, retries=3, retry_delay=5, retry_max_delay=1000)
        self.assertEqual(state["calls"], 3)
        # Backoff doubles: 5 -> 10.
        self.assertEqual(sleeps, [5, 10])

    def test_exhausts_retries_then_raises(self):
        stub, state = make_snapshot_download(5)
        with self.assertRaises(RuntimeError) as ctx:
            self.run_with(stub, retries=2, retry_delay=5, retry_max_delay=100)
        self.assertIn("after 3 attempts", str(ctx.exception))
        self.assertEqual(state["calls"], 3)

    def test_retries_zero_single_attempt(self):
        stub, state = make_snapshot_download(1)
        with self.assertRaises(RuntimeError) as ctx:
            self.run_with(stub, retries=0)
        self.assertIn("after 1 attempt", str(ctx.exception))
        self.assertEqual(state["calls"], 1)

    def test_backoff_doubles_and_caps(self):
        stub, state = make_snapshot_download(10)  # fails beyond the budget
        with self.assertRaises(RuntimeError):
            self.run_with(stub, retries=4, retry_delay=60, retry_max_delay=100)
        self.assertEqual(state["calls"], 5)

    def test_backoff_doubles_per_retry(self):
        stub, _ = make_snapshot_download(4)
        sleeps = self.run_with(stub, retries=4, retry_delay=5, retry_max_delay=1000)
        # 4 failures then success: delays double 5 -> 10 -> 20 -> 40.
        self.assertEqual(sleeps, [5, 10, 20, 40])

    def test_backoff_caps_at_max_delay(self):
        stub, _ = make_snapshot_download(4)
        sleeps = self.run_with(stub, retries=4, retry_delay=60, retry_max_delay=100)
        # 60 doubles to 120 -> capped at 100; stays capped after that.
        self.assertEqual(sleeps, [60, 100, 100, 100])

    def test_unlimited_retries_until_success(self):
        stub, state = make_snapshot_download(4)
        sleeps = self.run_with(stub, unlimited=True, retry_delay=10, retry_max_delay=40)
        self.assertEqual(state["calls"], 5)
        # 10 -> 20 -> 40 -> 40 (doubling capped at max).
        self.assertEqual(sleeps, [10, 20, 40, 40])

    def test_limited_retries_bounds_work(self):
        # Even in unlimited mode a permanently-failing download is only bounded
        # when we run a finite budget for the test; verify the strict budget.
        stub, state = make_snapshot_download(10 ** 9)
        with self.assertRaises(RuntimeError):
            self.run_with(stub, retries=5, retry_delay=1)
        self.assertEqual(state["calls"], 6)


if __name__ == "__main__":
    unittest.main()
