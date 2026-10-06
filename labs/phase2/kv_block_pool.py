"""Deterministic CPU teaching model for paged KV-cache blocks.

The real serving systems in the KV chapter use GPU memory, asynchronous kernels,
and a considerably richer scheduler.  This module intentionally models one thing
well: the ownership protocol around a logical block table.  A request owns a list
of logical blocks, each entry pointing at a physical block in a finite pool.
Physical blocks carry a reference count, so sharing a prefix is cheap and a write
to a shared block uses copy-on-write (COW).  A small prefix cache is keyed by a
stable SHA-256 digest and eviction is deterministic LRU over cache entries and
unreferenced blocks.

All state transitions are synchronous and use only the Python standard library.
The API returns copies rather than internal mutable containers, making examples
and tests reproducible.  It is a teaching model, not a GPU allocator or a claim
about production throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Public data records


@dataclass(frozen=True)
class PrefixEntry:
    """A cached prefix and the physical blocks backing it.

    ``physical_blocks`` are held by the cache itself.  A request attaching the
    entry adds another reference to every block; dropping/evicting the entry
    releases only the cache's references.
    """

    prefix_hash: str
    token_count: int
    tokens: tuple[int, ...]
    physical_blocks: tuple[int, ...]
    last_access: int

    @property
    def block_table(self) -> tuple[int, ...]:
        """Alias used in the chapter's logical-to-physical diagrams."""
        return self.physical_blocks


@dataclass
class PhysicalBlock:
    """Bookkeeping for one fixed-size physical block."""

    block_id: int
    tokens: list[int]
    refcount: int = 0
    last_access: int = 0

    @property
    def is_free(self) -> bool:
        return self.refcount == 0


@dataclass
class RequestState:
    """A request's logical block table and visible token length."""

    request_id: str
    logical_to_physical: list[int] = field(default_factory=list)
    token_count: int = 0
    prefix_hash: str | None = None
    last_access: int = 0

    @property
    def block_table(self) -> list[int]:
        """A familiar name for the logical -> physical table."""
        return self.logical_to_physical

    @property
    def num_blocks(self) -> int:
        return len(self.logical_to_physical)


# ``RequestTable`` is the chapter-facing name; it intentionally remains a
# lightweight record rather than exposing the pool's mutable internal object.
RequestTable = RequestState


class KVBlockPool:
    """Finite fixed-size physical block pool with sharing and COW.

    Parameters
    ----------
    capacity:
        Number of physical blocks.  IDs are allocated deterministically from
        ``0`` through ``capacity - 1``.
    block_size:
        Number of token slots in each block.  A request's final block may be
        partially full; ``RequestState.token_count`` says which slots are live.

    Notes
    -----
    A block's ``refcount`` includes both request-table references and references
    held by prefix-cache entries.  ``release_request`` and ``drop_prefix`` make
    those references explicit.  ``evict`` first drops old cache entries when
    needed, then frees zero-reference blocks.  No live request is evicted.
    """

    def __init__(
        self,
        capacity: int | None = None,
        block_size: int = 4,
        *,
        num_blocks: int | None = None,
        num_physical_blocks: int | None = None,
    ) -> None:
        # ``num_blocks`` is a readable spelling for chapter diagrams; keep
        # ``capacity`` as the positional form used by the lab itself.
        aliases = [value for value in (num_blocks, num_physical_blocks) if value is not None]
        if capacity is None:
            if not aliases:
                raise TypeError("capacity (or num_blocks) is required")
            capacity = aliases[0]
        if aliases and any(value != capacity for value in aliases):
            raise ValueError("capacity and block-count aliases disagree")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.capacity = capacity
        self.block_size = block_size
        self._blocks: dict[int, PhysicalBlock] = {}
        self._free: set[int] = set(range(capacity))
        self._requests: dict[str, RequestState] = {}
        self._prefixes: dict[str, PrefixEntry] = {}
        self._clock = 0
        self._cow_copies = 0
        self._hit_tokens = 0
        self._eviction_count = 0

    # ---- deterministic validation and bookkeeping ----------------------

    @staticmethod
    def _request_key(request_id: str) -> str:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        return request_id

    @staticmethod
    def _tokens(tokens: Iterable[int], *, name: str = "tokens") -> tuple[int, ...]:
        try:
            raw = tuple(tokens)
        except TypeError as exc:
            raise ValueError(f"{name} must be an iterable of integers") from exc
        if any(isinstance(token, bool) for token in raw):
            raise ValueError(f"{name} must contain integers, not bool")
        try:
            values = tuple(int(token) for token in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an iterable of integers") from exc
        return values

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _touch_block(self, block_id: int) -> None:
        block = self._blocks[block_id]
        block.last_access = self._tick()

    def _touch_request(self, request_id: str) -> RequestState:
        request = self._requests[request_id]
        request.last_access = self._tick()
        return request

    def _touch_prefix(self, prefix_hash: str) -> PrefixEntry:
        old = self._prefixes[prefix_hash]
        new = PrefixEntry(
            old.prefix_hash,
            old.token_count,
            old.tokens,
            old.physical_blocks,
            self._tick(),
        )
        self._prefixes[prefix_hash] = new
        for block_id in old.physical_blocks:
            self._touch_block(block_id)
        return new

    def _check_block_id(self, block_id: int) -> None:
        if not isinstance(block_id, int) or isinstance(block_id, bool) or not 0 <= block_id < self.capacity:
            raise ValueError("block_id is outside the pool")
        if block_id not in self._blocks:
            raise KeyError(f"physical block {block_id} is not allocated")

    def _ensure_capacity(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > self.capacity:
            raise MemoryError("requested allocation exceeds total pool capacity")
        while len(self._free) < count:
            # Reclaim as much as possible without surprising callers by
            # evicting a live request.  Prefix cache entries are evicted only
            # if all their references can be released safely.
            before = len(self._free)
            self.evict(count=count - len(self._free))
            if len(self._free) == before:
                raise MemoryError(
                    f"KV block pool exhausted: need {count} free blocks, "
                    f"have {len(self._free)}"
                )

    def _allocate(self, initial_tokens: Sequence[int] = ()) -> int:
        values = list(initial_tokens)
        if len(values) > self.block_size:
            raise ValueError("initial_tokens exceeds block_size")
        self._ensure_capacity(1)
        block_id = min(self._free)
        self._free.remove(block_id)
        self._blocks[block_id] = PhysicalBlock(block_id, values, refcount=0, last_access=self._tick())
        return block_id

    def _inc_ref(self, block_id: int, amount: int = 1) -> None:
        self._check_block_id(block_id)
        if amount < 0:
            raise ValueError("amount must be non-negative")
        block = self._blocks[block_id]
        block.refcount += amount
        self._touch_block(block_id)

    def _dec_ref(self, block_id: int, amount: int = 1) -> None:
        self._check_block_id(block_id)
        if amount < 0:
            raise ValueError("amount must be non-negative")
        block = self._blocks[block_id]
        if block.refcount < amount:
            raise RuntimeError(f"refcount underflow for physical block {block_id}")
        block.refcount -= amount
        self._touch_block(block_id)

    def _free_zero_blocks(self) -> list[int]:
        removed: list[int] = []
        for block_id in sorted(tuple(self._blocks)):
            block = self._blocks[block_id]
            if block.refcount == 0:
                removed.append(block_id)
                del self._blocks[block_id]
                self._free.add(block_id)
        return removed

    # ---- prefix hashing and cache ---------------------------------------

    @staticmethod
    def hash_prefix(tokens: Sequence[int], *, namespace: str = "kv-toy-v1") -> str:
        """Return a stable hash for a token prefix.

        JSON with fixed separators is deliberately used instead of Python's
        process-randomized ``hash``.  The namespace makes the key safe to
        invalidate when the tokenization or cache-key format changes.
        """
        raw = tuple(tokens)
        if any(isinstance(token, bool) for token in raw):
            raise ValueError("prefix tokens must contain integers, not bool")
        try:
            values = tuple(int(token) for token in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("prefix tokens must be an iterable of integers") from exc
        payload = json.dumps(
            {"namespace": namespace, "tokens": list(values)},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # Friendly names used by notebooks and tests.
    prefix_hash = hash_prefix
    compute_prefix_hash = hash_prefix

    def cache_prefix(
        self,
        tokens: Sequence[int] | None = None,
        *,
        request_id: str | None = None,
        token_count: int | None = None,
        namespace: str = "kv-toy-v1",
    ) -> str:
        """Store a prefix in the cache and return its digest.

        The prefix can be copied from an existing request (avoiding a second
        allocation) or built from token values directly.  Cache entries hold
        one reference to each listed block.  Prefixes may end in a partial
        block; a later append will COW that block when it is shared.
        """
        if request_id is not None:
            self._request_key(request_id)
            if request_id not in self._requests:
                raise KeyError(f"unknown request {request_id!r}")
            request = self._touch_request(request_id)
            if token_count is None:
                token_count = request.token_count
            if not isinstance(token_count, int) or isinstance(token_count, bool) or not 0 <= token_count <= request.token_count:
                raise ValueError("token_count must be within the request's live token range")
            request_tokens = tuple(self._request_tokens(request))
            values = request_tokens[:token_count]
            block_count = self._blocks_for_tokens(token_count)
            physical = tuple(request.logical_to_physical[:block_count])
        else:
            if tokens is None:
                raise ValueError("tokens or request_id is required")
            values = self._tokens(tokens)
            if token_count is not None:
                if not isinstance(token_count, int) or isinstance(token_count, bool) or not 0 <= token_count <= len(values):
                    raise ValueError("token_count must be within tokens")
                values = values[:token_count]
            physical = tuple()

        digest = self.hash_prefix(values, namespace=namespace)
        existing = self._prefixes.get(digest)
        if existing is not None:
            # A hash collision or namespace bug must never silently alias data.
            if existing.tokens != values:
                raise RuntimeError("prefix hash collision detected")
            self._touch_prefix(digest)
            return digest

        if request_id is None:
            physical = self._allocate_token_blocks(values)

        # A request-derived entry increments refs.  Fresh blocks currently have
        # zero refs, so both paths can use the same increment operation.
        for block_id in physical:
            self._inc_ref(block_id)
        entry = PrefixEntry(digest, len(values), values, physical, self._tick())
        self._prefixes[digest] = entry
        return digest

    def _allocate_token_blocks(self, tokens: Sequence[int]) -> tuple[int, ...]:
        if not tokens:
            return tuple()
        # Reserve enough space before publishing any zero-ref build blocks.
        # Allocation is synchronous, so no eviction can reclaim our own
        # unfinished allocation in the following loop.
        self._ensure_capacity(self._blocks_for_tokens(len(tokens)))
        block_ids: list[int] = []
        try:
            for start in range(0, len(tokens), self.block_size):
                block_ids.append(self._allocate(tokens[start : start + self.block_size]))
            return tuple(block_ids)
        except Exception:
            for block_id in block_ids:
                # They have zero refs; this is safe and deterministic.
                self._blocks.pop(block_id, None)
                self._free.add(block_id)
            raise

    def lookup_prefix(self, tokens: Sequence[int], *, namespace: str = "kv-toy-v1") -> PrefixEntry | None:
        """Return an immutable cache record for an exact prefix, if present."""
        values = self._tokens(tokens)
        digest = self.hash_prefix(values, namespace=namespace)
        entry = self._prefixes.get(digest)
        if entry is None or entry.tokens != values:
            return None
        return self._touch_prefix(digest)

    get_prefix = lookup_prefix

    def longest_prefix(
        self,
        tokens: Sequence[int],
        *,
        namespace: str = "kv-toy-v1",
    ) -> PrefixEntry | None:
        """Return the longest cached prefix of ``tokens``.

        Ties are resolved by digest, so a snapshot is reproducible even if a
        caller inserted equivalent entries in a different order.
        """
        values = self._tokens(tokens)
        candidates = [
            entry
            for entry in self._prefixes.values()
            if entry.token_count <= len(values)
            and entry.tokens == values[: entry.token_count]
            and self.hash_prefix(entry.tokens, namespace=namespace) == entry.prefix_hash
        ]
        if not candidates:
            return None
        selected = max(candidates, key=lambda entry: (entry.token_count, entry.prefix_hash))
        return self._touch_prefix(selected.prefix_hash)

    lookup_longest_prefix = longest_prefix

    def drop_prefix(self, prefix: str | Sequence[int], *, namespace: str = "kv-toy-v1") -> bool:
        """Remove one cache entry and release its references.

        ``prefix`` may be a digest or the original token sequence.  Returns
        ``False`` when no matching entry exists.
        """
        digest = prefix if isinstance(prefix, str) else self.hash_prefix(prefix, namespace=namespace)
        entry = self._prefixes.pop(digest, None)
        if entry is None:
            return False
        for block_id in entry.physical_blocks:
            self._dec_ref(block_id)
        self._free_zero_blocks()
        return True

    release_prefix = drop_prefix

    # ---- request lifecycle and COW --------------------------------------

    def _blocks_for_tokens(self, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        return (token_count + self.block_size - 1) // self.block_size

    def _request_tokens(self, request: RequestState) -> list[int]:
        values: list[int] = []
        for block_id in request.logical_to_physical:
            self._check_block_id(block_id)
            values.extend(self._blocks[block_id].tokens)
        return values[: request.token_count]

    def create_request(
        self,
        request_id: str,
        tokens: Sequence[int] = (),
        *,
        use_prefix_cache: bool = True,
        namespace: str = "kv-toy-v1",
    ) -> RequestState:
        """Create a request, optionally attaching an exact cached prefix."""
        key = self._request_key(request_id)
        if key in self._requests:
            raise ValueError(f"request {key!r} already exists")
        values = self._tokens(tokens)
        request = RequestState(key)
        self._requests[key] = request
        try:
            attached = False
            if use_prefix_cache and values:
                entry = self.longest_prefix(values, namespace=namespace)
                if entry is not None:
                    self._attach_entry(request, entry)
                    self._hit_tokens += entry.token_count
                    # The cache may contain a shorter prefix.  Only materialize
                    # the uncached suffix; replaying the whole input here would
                    # duplicate the cached tokens and obscure the block-table
                    # sharing lesson.
                    self._append_to_request(request, values[entry.token_count :])
                    attached = True
            if not attached:
                self._append_to_request(request, values)
            self._touch_request(key)
            return self.request_state(key)
        except Exception:
            self._requests.pop(key, None)
            self._release_mapping(request.logical_to_physical)
            raise

    new_request = create_request

    def _attach_entry(self, request: RequestState, entry: PrefixEntry) -> None:
        request.logical_to_physical = list(entry.physical_blocks)
        request.token_count = entry.token_count
        request.prefix_hash = entry.prefix_hash
        for block_id in request.logical_to_physical:
            self._inc_ref(block_id)
        self._touch_prefix(entry.prefix_hash)

    def attach_prefix(self, request_id: str, tokens: Sequence[int], *, namespace: str = "kv-toy-v1") -> bool:
        """Attach an exact cached prefix to a new or empty request.

        Returns ``True`` on a cache hit.  Existing non-empty requests are
        rejected to prevent an accidental table overwrite.
        """
        key = self._request_key(request_id)
        entry = self.lookup_prefix(tokens, namespace=namespace)
        if entry is None:
            return False
        if key in self._requests:
            request = self._requests[key]
            if request.token_count or request.logical_to_physical:
                raise ValueError("attach_prefix requires a new or empty request")
        else:
            self._requests[key] = RequestState(key)
        request = self._requests[key]
        self._attach_entry(request, entry)
        self._touch_request(key)
        return True

    restore_prefix = attach_prefix

    def _append_to_request(self, request: RequestState, values: Sequence[int]) -> None:
        for token in values:
            if not request.logical_to_physical or request.token_count % self.block_size == 0:
                block_id = self._allocate([token])
                request.logical_to_physical.append(block_id)
                self._inc_ref(block_id)
            else:
                logical = len(request.logical_to_physical) - 1
                block_id = self.ensure_writable(request.request_id, logical, _request=request)
                block = self._blocks[block_id]
                offset = request.token_count % self.block_size
                # A truncated fork can share a block whose hidden tail still
                # contains the parent's tokens.  Writing at the logical offset
                # must replace that tail, rather than append after it.
                if offset < len(block.tokens):
                    block.tokens[offset] = int(token)
                else:
                    block.tokens.append(int(token))
                self._touch_block(block_id)
            request.token_count += 1

    def append_tokens(self, request_id: str, tokens: Sequence[int]) -> RequestState:
        """Append tokens, COWing a shared final block before it is written."""
        key = self._request_key(request_id)
        if key not in self._requests:
            raise KeyError(f"unknown request {key!r}")
        values = self._tokens(tokens)
        request = self._touch_request(key)
        self._append_to_request(request, values)
        request.prefix_hash = None
        return self.request_state(key)

    append = append_tokens

    def append_token(self, request_id: str, token: int) -> RequestState:
        return self.append_tokens(request_id, [token])

    def ensure_writable(
        self,
        request_id: str,
        logical_index: int,
        *,
        _request: RequestState | None = None,
    ) -> int:
        """Return a writable physical block, copying it when shared."""
        key = self._request_key(request_id)
        request = _request if _request is not None else self._requests.get(key)
        if request is None:
            raise KeyError(f"unknown request {key!r}")
        if not isinstance(logical_index, int) or isinstance(logical_index, bool):
            raise ValueError("logical_index must be an integer")
        if not 0 <= logical_index < len(request.logical_to_physical):
            raise IndexError("logical_index is outside the request table")
        old_id = request.logical_to_physical[logical_index]
        self._check_block_id(old_id)
        old = self._blocks[old_id]
        if old.refcount <= 1:
            self._touch_block(old_id)
            return old_id
        new_id = self._allocate(old.tokens)
        # The new block belongs to this request before the old reference is
        # released; this ordering keeps the invariant visible to snapshots.
        self._inc_ref(new_id)
        self._dec_ref(old_id)
        request.logical_to_physical[logical_index] = new_id
        request.prefix_hash = None
        self._cow_copies += 1
        self._touch_request(key)
        return new_id

    copy_on_write = ensure_writable

    def fork_request(
        self,
        parent_id: str,
        child_id: str,
        *,
        token_count: int | None = None,
    ) -> RequestState:
        """Fork a request by sharing its block table; later writes COW."""
        parent_key = self._request_key(parent_id)
        child_key = self._request_key(child_id)
        if parent_key not in self._requests:
            raise KeyError(f"unknown request {parent_key!r}")
        if child_key in self._requests:
            raise ValueError(f"request {child_key!r} already exists")
        parent = self._touch_request(parent_key)
        length = parent.token_count if token_count is None else token_count
        if not isinstance(length, int) or isinstance(length, bool) or not 0 <= length <= parent.token_count:
            raise ValueError("token_count must be within the parent request")
        block_count = self._blocks_for_tokens(length)
        child = RequestState(child_key, list(parent.logical_to_physical[:block_count]), length)
        self._requests[child_key] = child
        for block_id in child.logical_to_physical:
            self._inc_ref(block_id)
        self._touch_request(child_key)
        return self.request_state(child_key)

    clone_request = fork_request

    allocate_request = create_request

    def share_prefix(
        self,
        request_id: str,
        prefix_tokens: Sequence[int] | str | None = None,
        *,
        namespace: str = "kv-toy-v1",
    ) -> str:
        """Publish a request prefix in the index and return its digest.

        If ``request_id`` is new, ``prefix_tokens`` is cached and attached to
        that request.  For an existing request, the live prefix (or the
        requested token count) is indexed without changing its table.
        """
        key = self._request_key(request_id)
        if isinstance(prefix_tokens, str):
            entry = self._prefixes.get(prefix_tokens)
            if entry is None:
                raise KeyError(f"unknown prefix hash {prefix_tokens!r}")
            if key in self._requests and self._requests[key].logical_to_physical:
                raise ValueError("share_prefix requires an empty/new request for a prefix hash")
            if key not in self._requests:
                self._requests[key] = RequestState(key)
            self._attach_entry(self._requests[key], entry)
            self._touch_request(key)
            return entry.prefix_hash
        if key in self._requests:
            if prefix_tokens is None:
                return self.cache_prefix(request_id=key, namespace=namespace)
            values = self._tokens(prefix_tokens)
            request_values = self.get_tokens(key)
            if len(values) > len(request_values) or tuple(request_values[: len(values)]) != values:
                raise ValueError("prefix_tokens must be a prefix of the request")
            return self.cache_prefix(request_id=key, token_count=len(values), namespace=namespace)
        if prefix_tokens is None:
            raise ValueError("prefix_tokens is required for a new request")
        values = self._tokens(prefix_tokens)
        digest = self.cache_prefix(values, namespace=namespace)
        self.attach_prefix(key, values, namespace=namespace)
        return digest

    def _release_mapping(self, mapping: Iterable[int]) -> None:
        for block_id in mapping:
            if block_id in self._blocks:
                self._dec_ref(block_id)
        self._free_zero_blocks()

    def release_request(self, request_id: str) -> bool:
        key = self._request_key(request_id)
        request = self._requests.pop(key, None)
        if request is None:
            return False
        self._release_mapping(request.logical_to_physical)
        return True

    delete_request = release_request
    close_request = release_request

    def request_state(self, request_id: str) -> RequestState:
        key = self._request_key(request_id)
        if key not in self._requests:
            raise KeyError(f"unknown request {key!r}")
        request = self._requests[key]
        return RequestState(
            request.request_id,
            list(request.logical_to_physical),
            request.token_count,
            request.prefix_hash,
            request.last_access,
        )

    def get_tokens(self, request_id: str) -> list[int]:
        key = self._request_key(request_id)
        if key not in self._requests:
            raise KeyError(f"unknown request {key!r}")
        request = self._touch_request(key)
        for block_id in request.logical_to_physical:
            self._touch_block(block_id)
        return self._request_tokens(request)

    tokens = get_tokens

    def block_table(self, request_id: str) -> list[int]:
        return self.request_state(request_id).logical_to_physical

    logical_to_physical = block_table

    def read_block(self, block_id: int) -> list[int]:
        self._check_block_id(block_id)
        self._touch_block(block_id)
        return list(self._blocks[block_id].tokens)

    def refcount(self, block_id: int) -> int:
        self._check_block_id(block_id)
        return self._blocks[block_id].refcount

    @property
    def refcounts(self) -> dict[int, int]:
        """Copy of the physical-id -> reference-count map for teaching traces."""
        return {block_id: self._blocks[block_id].refcount for block_id in sorted(self._blocks)}

    @property
    def physical_blocks(self) -> dict[int, list[int]]:
        """Copy of allocated physical block contents, keyed by block id."""
        return {block_id: list(self._blocks[block_id].tokens) for block_id in sorted(self._blocks)}

    # ---- deterministic eviction and snapshots --------------------------

    def _evict_cache_entry(self, digest: str) -> None:
        entry = self._prefixes.pop(digest)
        for block_id in entry.physical_blocks:
            self._dec_ref(block_id)

    def evict(
        self,
        count: int | None = None,
        *,
        target_free: int | None = None,
    ) -> list[int]:
        """Evict old cache entries and free blocks deterministically.

        ``count`` is a minimum number of physical blocks to return to the free
        pool, when possible. One cache entry may release several blocks, and
        all freed IDs are returned. ``target_free`` requests a free watermark.
        In either mode
        cache entries are considered oldest-first (last-access tick, digest),
        then zero-ref blocks are reclaimed in block-id order.  Active request
        references are never dropped.
        """
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
            raise ValueError("count must be a non-negative integer or None")
        if target_free is not None and (
            not isinstance(target_free, int) or isinstance(target_free, bool) or target_free < 0
        ):
            raise ValueError("target_free must be a non-negative integer or None")
        if count is None and target_free is None:
            target_free = self.capacity
        if target_free is not None:
            target_free = min(target_free, self.capacity)

        removed: list[int] = []
        def enough() -> bool:
            if target_free is not None and len(self._free) < target_free:
                return False
            return count is None or len(removed) >= count

        # Drop stale cache entries first.  This makes eviction useful even when
        # every physical block is pinned solely by the prefix cache.
        for entry in sorted(self._prefixes.values(), key=lambda e: (e.last_access, e.prefix_hash)):
            if enough():
                break
            self._evict_cache_entry(entry.prefix_hash)
            newly_free = self._free_zero_blocks()
            self._eviction_count += len(newly_free)
            for block_id in newly_free:
                if block_id not in removed:
                    removed.append(block_id)
        if not enough():
            newly_free = self._free_zero_blocks()
            self._eviction_count += len(newly_free)
            for block_id in newly_free:
                if block_id not in removed:
                    removed.append(block_id)
        # For count=0 we intentionally return no blocks and leave cache state
        # alone.  The branch above uses ``enough`` before any mutation.
        return sorted(removed)

    def stats(self) -> dict[str, Any]:
        allocated = len(self._blocks)
        referenced = sum(block.refcount for block in self._blocks.values())
        return {
            "capacity": self.capacity,
            "block_size": self.block_size,
            "allocated_blocks": allocated,
            "free_blocks": self.capacity - allocated,
            "referenced_blocks": sum(block.refcount > 0 for block in self._blocks.values()),
            "total_references": referenced,
            "requests": len(self._requests),
            "prefix_entries": len(self._prefixes),
        }

    def snapshot_metrics(self) -> dict[str, Any]:
        """Return lab metrics for one synchronous state-machine snapshot."""
        request_blocks = [
            block_id
            for request in self._requests.values()
            for block_id in request.logical_to_physical
        ]
        logical_tokens = sum(request.token_count for request in self._requests.values())
        physical_slots = len(request_blocks) * self.block_size
        active_blocks = len(set(request_blocks))
        waste = (physical_slots - logical_tokens) / physical_slots if physical_slots else 0.0
        return {
            "active_requests": len(self._requests),
            "active_blocks": active_blocks,
            "free_blocks": len(self._free),
            "logical_tokens": logical_tokens,
            "physical_slots": physical_slots,
            "internal_waste_ratio": waste,
            "hit_tokens": self._hit_tokens,
            "cow_copies": self._cow_copies,
            "evictions": self._eviction_count,
            "eviction_count": self._eviction_count,
            "prefix_entries": len(self._prefixes),
        }

    metrics = snapshot_metrics

    def snapshot(self) -> dict[str, Any]:
        """Return a stable JSON-compatible view for lab artifacts."""
        return {
            "capacity": self.capacity,
            "block_size": self.block_size,
            "clock": self._clock,
            "free_blocks": sorted(self._free),
            "blocks": {
                str(block_id): {
                    "tokens": list(self._blocks[block_id].tokens),
                    "refcount": self._blocks[block_id].refcount,
                    "last_access": self._blocks[block_id].last_access,
                }
                for block_id in sorted(self._blocks)
            },
            "requests": {
                request_id: {
                    "block_table": list(request.logical_to_physical),
                    "token_count": request.token_count,
                    "prefix_hash": request.prefix_hash,
                    "last_access": request.last_access,
                }
                for request_id, request in sorted(self._requests.items())
            },
            "prefix_cache": {
                digest: {
                    "token_count": entry.token_count,
                    "tokens": list(entry.tokens),
                    "physical_blocks": list(entry.physical_blocks),
                    "last_access": entry.last_access,
                }
                for digest, entry in sorted(self._prefixes.items())
            },
        }

    describe = snapshot

    def assert_invariants(self) -> None:
        """Raise ``AssertionError`` if ownership or table invariants break."""
        assert self._free.isdisjoint(self._blocks)
        assert self._free | set(self._blocks) == set(range(self.capacity))
        expected: dict[int, int] = {block_id: 0 for block_id in self._blocks}
        for request in self._requests.values():
            assert len(request.logical_to_physical) == self._blocks_for_tokens(request.token_count)
            for block_id in request.logical_to_physical:
                assert block_id in self._blocks
                expected[block_id] += 1
        for entry in self._prefixes.values():
            assert len(entry.physical_blocks) == self._blocks_for_tokens(entry.token_count)
            for block_id in entry.physical_blocks:
                assert block_id in self._blocks
                expected[block_id] += 1
        for block_id, block in self._blocks.items():
            assert block.refcount == expected[block_id], (block_id, block.refcount, expected[block_id])
            assert len(block.tokens) <= self.block_size


# Names used in prose and simple notebooks.  The chapter spells the compact
# constructor as ``BlockPool(block_size, capacity)``; the core class keeps the
# less ambiguous ``KVBlockPool(capacity, block_size)`` positional form.
class BlockPool(KVBlockPool):
    def __init__(self, block_size: int = 4, capacity: int = 64, **kwargs: Any) -> None:
        super().__init__(capacity=capacity, block_size=block_size, **kwargs)


KVCache = KVBlockPool


class PrefixIndex:
    """Small adapter exposing the prefix-index operations as an object."""

    def __init__(self, pool: KVBlockPool) -> None:
        if not isinstance(pool, KVBlockPool):
            raise TypeError("PrefixIndex requires a KVBlockPool")
        self.pool = pool

    def insert(self, tokens: Sequence[int], *, request_id: str | None = None) -> str:
        return self.pool.cache_prefix(tokens, request_id=request_id)

    def lookup(self, tokens: Sequence[int]) -> PrefixEntry | None:
        return self.pool.lookup_prefix(tokens)

    def longest(self, tokens: Sequence[int]) -> PrefixEntry | None:
        return self.pool.longest_prefix(tokens)


# Functional spelling for tiny notebooks that do not need a pool instance.
prefix_hash = KVBlockPool.hash_prefix
compute_prefix_hash = KVBlockPool.hash_prefix


def run_demo() -> dict[str, Any]:
    """Run a tiny sharing/COW/eviction trace for an optional JSON artifact."""
    pool = KVBlockPool(capacity=6, block_size=2)
    pool.create_request("parent", [10, 11, 12])
    digest = pool.cache_prefix(request_id="parent")
    pool.fork_request("parent", "child")
    before = pool.snapshot()
    pool.append_token("child", 99)  # current partial block is shared -> COW
    after_cow = pool.snapshot()
    pool.release_request("parent")
    pool.release_request("child")
    pool.evict()
    return {
        "prefix_hash": digest,
        "before_cow": before,
        "after_cow": after_cow,
        "after_release_and_eviction": pool.snapshot(),
    }


def _linear_percentile(samples: Sequence[float], percentile: float) -> float:
    values = sorted(float(value) for value in samples)
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _summary(samples: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "mean": sum(values) / len(values),
        "p50": _linear_percentile(values, 50),
        "p95": _linear_percentile(values, 95),
        "p99": _linear_percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _finite_vector(values: Sequence[float], *, name: str) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector


def _attention(
    query: Sequence[float],
    keys: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    *,
    scale: float | None = None,
    mask: Sequence[bool] | None = None,
) -> list[float]:
    """Shared numerically stable attention core used by dense and paged paths."""
    import math

    q = _finite_vector(query, name="query")
    key_rows = [_finite_vector(row, name="key row") for row in keys]
    value_rows = [_finite_vector(row, name="value row") for row in values]
    if not key_rows or len(key_rows) != len(value_rows):
        raise ValueError("keys and values must have the same non-zero row count")
    if any(len(row) != len(q) for row in key_rows):
        raise ValueError("every key row must match query width")
    value_width = len(value_rows[0])
    if any(len(row) != value_width for row in value_rows):
        raise ValueError("every value row must have the same width")
    if mask is None:
        allowed = [True] * len(key_rows)
    else:
        allowed = [bool(item) for item in mask]
        if len(allowed) != len(key_rows):
            raise ValueError("mask length must match sequence length")
    if not any(allowed):
        raise ValueError("attention mask selects no rows")
    factor = 1.0 / math.sqrt(len(q)) if scale is None else float(scale)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("scale must be finite and > 0")
    logits = [
        math.fsum(q_i * k_i for q_i, k_i in zip(q, key_row)) * factor if keep else -math.inf
        for key_row, keep in zip(key_rows, allowed)
    ]
    pivot = max(logit for logit in logits if logit != -math.inf)
    weights = [0.0 if logit == -math.inf else math.exp(logit - pivot) for logit in logits]
    denominator = math.fsum(weights)
    weights = [weight / denominator for weight in weights]
    return [
        math.fsum(weight * row[column] for weight, row in zip(weights, value_rows))
        for column in range(value_width)
    ]


def dense_attention(
    query: Sequence[float],
    keys: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    *,
    scale: float | None = None,
    mask: Sequence[bool] | None = None,
) -> list[float]:
    """Reference attention over contiguous key/value rows (CPU only)."""
    return _attention(query, keys, values, scale=scale, mask=mask)


def paged_attention(
    query: Sequence[float],
    key_blocks: Mapping[int, Sequence[Sequence[float]]],
    block_table: Sequence[int],
    seq_len: int,
    *,
    value_blocks: Mapping[int, Sequence[Sequence[float]]] | None = None,
    values: Mapping[int, Sequence[Sequence[float]]] | None = None,
    scale: float | None = None,
    mask: Sequence[bool] | None = None,
) -> list[float]:
    """Read non-contiguous physical blocks through a logical block table.

    ``seq_len`` is the logical valid-token count.  Rows after that boundary,
    including stale rows in a partially filled final block, are never read.
    The function intentionally materializes rows before invoking the dense core
    so tests can prove that address translation changes no attention math.
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if not block_table:
        raise ValueError("block_table must not be empty")
    if value_blocks is not None and values is not None:
        raise ValueError("pass only one of value_blocks or values")
    values_source = key_blocks if value_blocks is None and values is None else (value_blocks or values)
    key_rows: list[Sequence[float]] = []
    value_rows: list[Sequence[float]] = []
    for block_id in block_table:
        if block_id not in key_blocks or block_id not in values_source:
            raise KeyError(f"block {block_id!r} is missing from the physical pool")
        key_block = key_blocks[block_id]
        value_block = values_source[block_id]
        if len(key_block) != len(value_block):
            raise ValueError(f"key/value block {block_id!r} row counts differ")
        key_rows.extend(key_block)
        value_rows.extend(value_block)
        if len(key_rows) >= seq_len:
            break
    if len(key_rows) < seq_len:
        raise ValueError("seq_len exceeds rows addressed by block_table")
    logical_keys = key_rows[:seq_len]
    logical_values = value_rows[:seq_len]
    logical_mask = None if mask is None else list(mask)[:seq_len]
    if mask is not None and len(mask) < seq_len:
        raise ValueError("mask length must cover seq_len")
    return _attention(query, logical_keys, logical_values, scale=scale, mask=logical_mask)


def _benchmark_repetition(
    *,
    seed: int,
    requests: int,
    block_size: int,
    capacity: int,
    prefix_rate: float,
) -> dict[str, float | int | bool]:
    rng = random.Random(seed)
    pool = KVBlockPool(capacity=capacity, block_size=block_size)
    # One complete, immutable prefix supplies a deterministic cache hit.  Its
    # key is only token ids in this toy; production keys also include model,
    # tokenizer, LoRA and tenant/ACL namespaces.
    prefix = tuple(101 + index for index in range(block_size * 2))
    pool.create_request("__seed__", prefix, use_prefix_cache=False)
    pool.cache_prefix(request_id="__seed__")
    pool.release_request("__seed__")

    lengths = (3, 7, 15, 16, 17, 31, 32, 33, 63)
    operation_start = time.perf_counter_ns()
    hit_tokens = 0
    logical_tokens = 0
    aggregate_physical_slots = 0
    aggregate_logical_tokens = 0
    peak_active_blocks = 0
    for index in range(requests):
        length = lengths[rng.randrange(len(lengths))]
        use_hit = rng.random() < prefix_rate and length >= len(prefix)
        if use_hit:
            suffix_length = length - len(prefix)
            tokens = prefix + tuple(100000 + index * 1000 + j for j in range(suffix_length))
        else:
            tokens = tuple(200000 + index * 1000 + j for j in range(length))
        logical_tokens += len(tokens)
        request_id = f"request-{index}"
        expected_hit = pool.longest_prefix(tokens) if use_hit else None
        state = pool.create_request(request_id, tokens, use_prefix_cache=True)
        if expected_hit is not None:
            hit_tokens += expected_hit.token_count
        metrics = pool.snapshot_metrics()
        aggregate_physical_slots += int(metrics["physical_slots"])
        aggregate_logical_tokens += int(metrics["logical_tokens"])
        peak_active_blocks = max(peak_active_blocks, int(metrics["active_blocks"]))
        pool.release_request(request_id)

    # A one-token shared block makes COW observable even though the main prefix
    # cache uses complete blocks, matching the chapter's explicit COW probe.
    pool.create_request("__cow_a__", [1], use_prefix_cache=False)
    pool.fork_request("__cow_a__", "__cow_b__")
    pool.append_token("__cow_b__", 2)
    pool.release_request("__cow_a__")
    pool.release_request("__cow_b__")

    # Drop the cache, then allocate again to verify that eviction did not leave
    # a leaked reference or corrupt the free queue.
    pool.evict(target_free=capacity)
    pool.create_request("__recovery__", [9, 8], use_prefix_cache=False)
    recovery_ok = pool.get_tokens("__recovery__") == [9, 8]
    pool.release_request("__recovery__")
    elapsed_us = (time.perf_counter_ns() - operation_start) / 1000.0
    metrics = pool.snapshot_metrics()
    return {
        "operation_us": elapsed_us,
        "logical_tokens": logical_tokens,
        "internal_waste_ratio": (
            (aggregate_physical_slots - aggregate_logical_tokens) / aggregate_physical_slots
            if aggregate_physical_slots
            else 0.0
        ),
        "hit_tokens": hit_tokens,
        "peak_active_blocks": peak_active_blocks,
        "cow_copies": int(metrics["cow_copies"]),
        "evictions": int(metrics["evictions"]),
        "eviction_count": int(metrics["evictions"]),
        "eviction_recovery": bool(recovery_ok),
        "requests": requests,
    }


def run_benchmark(
    *,
    seed: int = 7,
    requests: int = 60,
    capacity: int = 64,
    repeats: int = 30,
    block_sizes: Sequence[int] = (4, 8, 16),
    prefix_rates: Sequence[float] = (0.0, 0.5, 1.0),
    warmup: int = 5,
) -> dict[str, Any]:
    """Run the chapter's L0 benchmark matrix and return a JSON artifact.

    The operation timer covers Python state transitions only.  It is useful for
    comparing this toy's control flow, but it is not GPU TTFT/TPOT evidence.
    """
    if requests <= 0 or capacity <= 0 or repeats <= warmup or warmup < 0:
        raise ValueError("requests/capacity must be positive and repeats > warmup")
    if not block_sizes or not prefix_rates:
        raise ValueError("benchmark matrix must not be empty")
    cases: list[dict[str, Any]] = []
    measured = repeats - warmup
    for block_size in block_sizes:
        for prefix_rate in prefix_rates:
            if not 0.0 <= float(prefix_rate) <= 1.0:
                raise ValueError("prefix_rates must be in [0, 1]")
            raw: list[dict[str, float | int | bool]] = []
            for repeat in range(repeats):
                sample = _benchmark_repetition(
                    seed=seed + repeat,
                    requests=requests,
                    block_size=int(block_size),
                    capacity=capacity,
                    prefix_rate=float(prefix_rate),
                )
                if repeat >= warmup:
                    raw.append(sample)
            metric_names = (
                "operation_us",
                "internal_waste_ratio",
                "hit_tokens",
                "peak_active_blocks",
                "cow_copies",
                "evictions",
                "eviction_count",
            )
            summary = {name: _summary([float(row[name]) for row in raw]) for name in metric_names}
            cases.append(
                {
                    "block_size": int(block_size),
                    "prefix_rate": float(prefix_rate),
                    "warmup_repeats": warmup,
                    "measured_repeats": measured,
                    "raw": raw,
                    "summary": summary,
                    "all_eviction_recoveries": all(bool(row["eviction_recovery"]) for row in raw),
                }
            )
    return {
        "level": "L0",
        "measurement": "CPU Python state transitions; operation_us is not GPU latency",
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "parameters": {
            "seed": seed,
            "requests": requests,
            "capacity": capacity,
            "repeats": repeats,
            "warmup": warmup,
            "block_sizes": [int(value) for value in block_sizes],
            "prefix_rates": [float(value) for value in prefix_rates],
            "request_lengths": [3, 7, 15, 16, 17, 31, 32, 33, 63],
        },
        "aggregation": {
            "percentile": "linear interpolation on sorted samples",
            "raw_samples_retained": True,
        },
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output", help="write the deterministic demo JSON to this path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--prefix-rate", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--report", help="write the benchmark JSON artifact to this path")
    args = parser.parse_args(argv)
    if args.report or any(
        value is not None
        for value in (args.seed, args.requests, args.block_size, args.capacity, args.prefix_rate, args.repeats)
    ):
        artifact = run_benchmark(
            seed=7 if args.seed is None else args.seed,
            requests=60 if args.requests is None else args.requests,
            capacity=64 if args.capacity is None else args.capacity,
            repeats=30 if args.repeats is None else args.repeats,
            block_sizes=(4, 8, 16) if args.block_size is None else (args.block_size,),
            prefix_rates=(0.0, 0.5, 1.0) if args.prefix_rate is None else (args.prefix_rate,),
        )
        encoded = json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")
        else:
            print(encoded)
        return 0
    artifact = run_demo()
    encoded = json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
