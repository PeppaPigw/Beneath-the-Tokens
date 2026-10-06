from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

try:  # Works both from ``labs/phase2`` and from repository root discovery.
    from kv_block_pool import KVBlockPool, KVCache, BlockPool, run_demo, run_benchmark, dense_attention, paged_attention
except ModuleNotFoundError:  # pragma: no cover - import-mode compatibility
    from labs.phase2.kv_block_pool import KVBlockPool, KVCache, BlockPool, run_demo, run_benchmark, dense_attention, paged_attention


class KVBlockPoolTests(unittest.TestCase):
    def test_duplicate_direct_prefix_insert_does_not_allocate_or_underflow(self) -> None:
        pool = KVBlockPool(3, block_size=2)
        first = pool.cache_prefix([1, 2, 3])
        table = pool.lookup_prefix([1, 2, 3]).physical_blocks
        second = pool.cache_prefix([1, 2, 3])
        self.assertEqual(first, second)
        self.assertEqual(pool.lookup_prefix([1, 2, 3]).physical_blocks, table)
        self.assertEqual(pool.stats()["allocated_blocks"], 2)
        pool.assert_invariants()

    def test_duplicate_request_prefix_insert_keeps_both_request_owners(self) -> None:
        pool = KVBlockPool(6, block_size=2)
        pool.create_request("a", [1, 2, 3], use_prefix_cache=False)
        pool.cache_prefix(request_id="a")
        pool.create_request("b", [1, 2, 3], use_prefix_cache=False)
        pool.cache_prefix(request_id="b")
        self.assertEqual(pool.get_tokens("b"), [1, 2, 3])
        pool.assert_invariants()

    def test_oversized_prefix_fails_without_recycling_its_own_build_blocks(self) -> None:
        pool = KVBlockPool(2, block_size=2)
        with self.assertRaises(MemoryError):
            pool.cache_prefix([1, 2, 3, 4, 5])
        self.assertEqual(pool.stats()["allocated_blocks"], 0)
        pool.assert_invariants()

    def test_eviction_reports_every_freed_block_when_entry_spans_blocks(self) -> None:
        pool = KVBlockPool(4, block_size=2)
        pool.cache_prefix([1, 2, 3, 4])
        self.assertEqual(pool.evict(count=1), [0, 1])
        self.assertEqual(pool.stats()["free_blocks"], 4)
        pool.assert_invariants()

    def test_logical_to_physical_table_and_block_boundaries(self) -> None:
        pool = KVBlockPool(capacity=4, block_size=2)
        state = pool.create_request("r", [10, 11, 12, 13, 14])
        self.assertEqual(state.block_table, [0, 1, 2])
        self.assertEqual(pool.get_tokens("r"), [10, 11, 12, 13, 14])
        self.assertEqual(pool.read_block(0), [10, 11])
        self.assertEqual(pool.read_block(2), [14])
        self.assertEqual(pool.stats()["allocated_blocks"], 3)
        pool.assert_invariants()

    def test_refcount_tracks_requests_and_release(self) -> None:
        pool = KVBlockPool(4, block_size=2)
        pool.create_request("parent", [1, 2, 3])
        parent_table = pool.block_table("parent")
        self.assertEqual([pool.refcount(i) for i in parent_table], [1, 1])
        pool.fork_request("parent", "child")
        self.assertEqual([pool.refcount(i) for i in parent_table], [2, 2])
        pool.release_request("parent")
        self.assertEqual([pool.refcount(i) for i in parent_table], [1, 1])
        pool.release_request("child")
        self.assertEqual(pool.stats()["allocated_blocks"], 0)
        self.assertEqual(pool.stats()["free_blocks"], 4)
        pool.assert_invariants()

    def test_copy_on_write_preserves_parent_when_child_appends_partial_block(self) -> None:
        pool = KVBlockPool(5, block_size=4)
        pool.create_request("parent", [1, 2, 3])
        parent_block = pool.block_table("parent")[0]
        pool.fork_request("parent", "child")
        pool.append_token("child", 99)
        child_block = pool.block_table("child")[0]
        self.assertNotEqual(parent_block, child_block)
        self.assertEqual(pool.get_tokens("parent"), [1, 2, 3])
        self.assertEqual(pool.get_tokens("child"), [1, 2, 3, 99])
        self.assertEqual(pool.refcount(parent_block), 1)
        self.assertEqual(pool.refcount(child_block), 1)
        pool.assert_invariants()

    def test_copy_on_write_after_truncated_fork_replaces_hidden_tail(self) -> None:
        pool = KVBlockPool(5, block_size=4)
        pool.create_request("parent", [1, 2, 3, 4])
        pool.fork_request("parent", "child", token_count=2)
        pool.append_tokens("child", [8, 9])
        self.assertEqual(pool.get_tokens("parent"), [1, 2, 3, 4])
        self.assertEqual(pool.get_tokens("child"), [1, 2, 8, 9])
        pool.assert_invariants()

    def test_prefix_hash_is_stable_and_cache_attach_shares_blocks(self) -> None:
        first = KVBlockPool.hash_prefix([1, 2, 3])
        second = KVBlockPool.hash_prefix(iter([1, 2, 3]))
        self.assertEqual(first, second)
        self.assertNotEqual(first, KVBlockPool.hash_prefix([1, 2, 4]))

        pool = KVBlockPool(6, block_size=2)
        pool.create_request("seed", [1, 2, 3])
        digest = pool.cache_prefix(request_id="seed")
        entry = pool.lookup_prefix([1, 2, 3])
        assert entry is not None
        self.assertEqual(entry.prefix_hash, digest)
        self.assertEqual(entry.block_table, tuple(pool.block_table("seed")))

        hit = pool.create_request("hit", [1, 2, 3, 4, 5])
        # The first full block is shared.  The cached final partial block is
        # COWed before the suffix is written, so it intentionally changes id.
        self.assertEqual(hit.block_table[0], entry.physical_blocks[0])
        self.assertEqual(pool.get_tokens("hit"), [1, 2, 3, 4, 5])
        # The suffix starts in a new block, because the cached final block was
        # partial and was COWed before token 4 was appended.
        self.assertNotEqual(pool.block_table("hit")[1], pool.block_table("seed")[1])
        pool.assert_invariants()

    def test_prefix_cache_reference_survives_request_release_then_drops(self) -> None:
        pool = KVBlockPool(4, block_size=2)
        pool.create_request("r", [7, 8, 9])
        digest = pool.cache_prefix(request_id="r")
        table = pool.block_table("r")
        pool.release_request("r")
        self.assertEqual([pool.refcount(block_id) for block_id in table], [1, 1])
        self.assertIsNotNone(pool.lookup_prefix([7, 8, 9]))
        self.assertTrue(pool.drop_prefix(digest))
        self.assertEqual(pool.stats()["allocated_blocks"], 0)
        self.assertFalse(pool.drop_prefix(digest))
        pool.assert_invariants()

    def test_longest_prefix_attach_replays_only_uncached_suffix(self) -> None:
        pool = KVBlockPool(8, block_size=2)
        pool.cache_prefix([1, 2])
        pool.cache_prefix([1, 2, 3, 4])
        state = pool.create_request("r", [1, 2, 3, 4, 5])
        self.assertEqual(state.token_count, 5)
        self.assertEqual(pool.get_tokens("r"), [1, 2, 3, 4, 5])
        self.assertEqual(state.block_table[:2], list(pool.lookup_prefix([1, 2, 3, 4]).physical_blocks))  # type: ignore[union-attr]
        pool.assert_invariants()

    def test_eviction_is_deterministic_and_never_drops_live_requests(self) -> None:
        pool = KVBlockPool(3, block_size=2)
        pool.create_request("live", [1, 2])
        pool.cache_prefix([3, 4])
        self.assertEqual(pool.stats()["allocated_blocks"], 2)
        removed = pool.evict(target_free=2)
        self.assertEqual(removed, [1])
        self.assertEqual(pool.get_tokens("live"), [1, 2])
        self.assertEqual(pool.stats()["free_blocks"], 2)
        self.assertIsNone(pool.lookup_prefix([3, 4]))
        # The live block remains pinned; a second eviction can release no more.
        self.assertEqual(pool.evict(target_free=3), [])
        pool.assert_invariants()

    def test_capacity_error_is_explicit_when_all_blocks_are_pinned(self) -> None:
        pool = KVBlockPool(1, block_size=2)
        pool.create_request("r", [1, 2])
        with self.assertRaises(MemoryError):
            pool.append_token("r", 3)
        self.assertEqual(pool.get_tokens("r"), [1, 2])
        pool.assert_invariants()

    def test_snapshot_is_json_safe_and_aliases_are_compatible(self) -> None:
        self.assertIs(KVCache, KVBlockPool)
        self.assertTrue(issubclass(BlockPool, KVBlockPool))
        chapter_pool = BlockPool(2, 3)
        self.assertEqual(chapter_pool.block_size, 2)
        self.assertEqual(chapter_pool.capacity, 3)
        pool = KVBlockPool(num_blocks=3, block_size=2)
        pool.create_request("r", [1])
        snapshot = pool.snapshot()
        encoded = json.dumps(snapshot, sort_keys=True)
        self.assertIn('"block_size": 2', encoded)
        self.assertEqual(snapshot, pool.describe())
        pool.assert_invariants()

    def test_demo_is_reproducible_and_writes_optional_artifact(self) -> None:
        first = run_demo()
        second = run_demo()
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kv-demo.json"
            # Keep CLI behavior covered without spawning a subprocess.
            path.write_text(json.dumps(first, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["prefix_hash"], first["prefix_hash"])

    def test_benchmark_matrix_retains_raw_samples_and_percentiles(self) -> None:
        report = run_benchmark(
            seed=7,
            requests=8,
            capacity=16,
            repeats=3,
            warmup=1,
            block_sizes=(4, 8),
            prefix_rates=(0.0, 1.0),
        )
        self.assertEqual(len(report["cases"]), 4)
        self.assertEqual(report["parameters"]["seed"], 7)
        for case in report["cases"]:
            self.assertEqual(len(case["raw"]), 2)
            self.assertEqual(case["summary"]["operation_us"]["n"], 2)
            self.assertIn("p95", case["summary"]["operation_us"])
            self.assertTrue(case["all_eviction_recoveries"])
        hit_case = next(case for case in report["cases"] if case["prefix_rate"] == 1.0 and case["block_size"] == 4)
        self.assertGreater(sum(row["hit_tokens"] for row in hit_case["raw"]), 0)

    def test_dense_and_paged_attention_match_noncontiguous_partial_blocks(self) -> None:
        query = [0.25, -0.5, 0.75]
        keys = [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [0.5, -1.0, 0.25],
            [-0.5, 0.25, 1.0],
            [0.25, 0.5, -1.0],
        ]
        values = [[float(i), float(i) + 0.5] for i in range(len(keys))]
        blocks = {7: (keys[2], keys[3]), 2: (keys[0], keys[1]), 5: (keys[4], [99.0, 99.0, 99.0])}
        value_blocks = {7: (values[2], values[3]), 2: (values[0], values[1]), 5: (values[4], [88.0, 88.5])}
        expected = dense_attention(query, keys, values)
        actual = paged_attention(query, blocks, [2, 7, 5], seq_len=5, value_blocks=value_blocks)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertAlmostEqual(left, right, places=12)

    def test_paged_attention_rejects_inconsistent_partial_length(self) -> None:
        query = [1.0, 0.0]
        blocks = {3: ([1.0, 0.0], [0.0, 1.0])}
        values = {3: ([1.0], [2.0])}
        with self.assertRaises(ValueError):
            paged_attention(query, blocks, [3], seq_len=3, value_blocks=values)

    def test_small_random_trace_preserves_invariants(self) -> None:
        rng = random.Random(7)
        pool = KVBlockPool(24, block_size=3)
        active = ["r0"]
        pool.create_request(active[0], [0, 1])
        for step in range(80):
            request_id = rng.choice(active)
            action = rng.randrange(4)
            if action < 2:
                pool.append_token(request_id, rng.randrange(100))
            elif action == 2 and len(active) < 6:
                child = f"r{step + 1}"
                if child in active:
                    continue
                pool.fork_request(request_id, child)
                active.append(child)
            elif len(active) > 1:
                doomed = active.pop(rng.randrange(1, len(active)))
                pool.release_request(doomed)
            pool.assert_invariants()


if __name__ == "__main__":
    unittest.main()
