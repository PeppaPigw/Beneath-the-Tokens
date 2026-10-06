"""Deterministic CPU reference implementations for chapter 23.

The functions here deliberately operate on Python lists and use only the standard
library.  They are teaching models: they preserve the ordering and normalization
rules that a production sampler needs, but they are not a benchmark for GPU code.
All functions return fresh lists and never mutate their input.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Sequence
from typing import Any


def _as_floats(values: Sequence[float], *, name: str = "values") -> list[float]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    result = [float(v) for v in values]
    if any(math.isnan(v) for v in result):
        raise ValueError(f"{name} contains NaN")
    return result


def normalize(probabilities: Sequence[float]) -> list[float]:
    """Return a probability vector with sum one, rejecting negatives."""
    values = _as_floats(probabilities, name="probabilities")
    if any(v < 0.0 or not math.isfinite(v) for v in values):
        raise ValueError("probabilities must be finite and non-negative")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    return [v / total for v in values]


def apply_temperature(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Scale logits by ``temperature``; T<1 sharpens and T>1 flattens."""
    values = _as_floats(logits, name="logits")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    # ``-inf`` is the conventional grammar/truncation mask.  Positive infinity
    # is rejected because it would make the softmax support ambiguous.
    if any(math.isnan(v) or v == math.inf for v in values):
        raise ValueError("logits must be finite except for -inf masks")
    return [v / temperature for v in values]


def logits_to_probabilities(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Stable softmax, optionally after temperature scaling."""
    scaled = apply_temperature(logits, temperature)
    finite = [v for v in scaled if v != -math.inf]
    if not finite:
        raise ValueError("at least one logit must be greater than -inf")
    pivot = max(finite)
    exp_values = [0.0 if v == -math.inf else math.exp(v - pivot) for v in scaled]
    return normalize(exp_values)


# Friendly aliases used in prose and by small notebooks.
softmax = logits_to_probabilities
logits_to_probs = logits_to_probabilities


def _renormalize_support(probabilities: Sequence[float], keep: Iterable[int]) -> list[float]:
    values = normalize(probabilities)
    indices = set(int(i) for i in keep)
    if not indices or min(indices) < 0 or max(indices) >= len(values):
        raise ValueError("support contains an invalid token index")
    mass = math.fsum(values[i] for i in indices)
    if mass <= 0.0:
        raise ValueError("selected support has zero probability mass")
    return [values[i] / mass if i in indices else 0.0 for i in range(len(values))]


def top_k(probabilities: Sequence[float], k: int) -> list[float]:
    """Keep the k largest probabilities and renormalize (a distribution change)."""
    values = normalize(probabilities)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(values):
        raise ValueError("k must be an integer in [1, vocabulary_size]")
    # Index is a deterministic tie-break: lower token id wins equal scores.
    keep = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    return _renormalize_support(values, keep)


def top_p(probabilities: Sequence[float], p: float) -> list[float]:
    """Keep the smallest descending prefix whose mass reaches p."""
    values = normalize(probabilities)
    if not math.isfinite(p) or not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    ordered = sorted(range(len(values)), key=lambda i: (-values[i], i))
    keep: list[int] = []
    mass = 0.0
    for index in ordered:
        keep.append(index)
        mass += values[index]
        if mass + 1e-15 >= p:
            break
    return _renormalize_support(values, keep)


def typical(probabilities: Sequence[float], typical_p: float) -> list[float]:
    """Locally typical sampling: retain surprisal closest to entropy."""
    values = normalize(probabilities)
    if not math.isfinite(typical_p) or not 0.0 < typical_p <= 1.0:
        raise ValueError("typical_p must be in (0, 1]")
    entropy = -math.fsum(p * math.log(p) for p in values if p > 0.0)
    ordered = sorted(
        range(len(values)),
        key=lambda i: (abs(-math.log(values[i]) - entropy) if values[i] > 0.0 else math.inf, i),
    )
    keep: list[int] = []
    mass = 0.0
    for index in ordered:
        if values[index] <= 0.0:
            continue
        keep.append(index)
        mass += values[index]
        if mass + 1e-15 >= typical_p:
            break
    return _renormalize_support(values, keep)


def min_p(probabilities: Sequence[float], minimum: float) -> list[float]:
    """Keep tokens with p(token) >= minimum * max(p), then renormalize."""
    values = normalize(probabilities)
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum must be in [0, 1]")
    threshold = minimum * max(values)
    keep = [i for i, value in enumerate(values) if value + 1e-15 >= threshold]
    return _renormalize_support(values, keep)


# Explicit ``*_filter``/``*_scale`` spellings make the teaching API easy to
# discover from a notebook while retaining the short names used in the text.
temperature_scale = apply_temperature
top_k_filter = top_k
top_p_filter = top_p
typical_filter = typical
min_p_filter = min_p


def entropy(probabilities: Sequence[float], *, base: float = math.e) -> float:
    """Shannon entropy in nats by default, or bits when ``base=2``."""
    values = normalize(probabilities)
    if base <= 0.0 or base == 1.0 or not math.isfinite(base):
        raise ValueError("base must be finite, positive, and != 1")
    value = -math.fsum(p * math.log(p) for p in values if p > 0.0)
    return value / math.log(base)


def mirostat_update(
    mu: float,
    surprisal: float,
    target_perplexity: float,
    eta: float = 0.1,
) -> float:
    """One Mirostat feedback update in log-perplexity coordinates.

    ``mu`` and ``log(target_perplexity)`` are nats/token.  Positive error means
    the observed token was too surprising, so the next threshold is lowered.
    This is the scalar control update, independent of the paper's Zipf tail
    estimator used to turn ``mu`` into a top-k value.
    """
    for name, value in (("mu", mu), ("surprisal", surprisal), ("target_perplexity", target_perplexity), ("eta", eta)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if target_perplexity <= 0.0 or eta < 0.0:
        raise ValueError("target_perplexity must be > 0 and eta must be >= 0")
    target_mu = math.log(target_perplexity)
    return float(mu) - float(eta) * (float(surprisal) - target_mu)


mirostat_step = mirostat_update


def grammar_filter(
    logits: Sequence[float],
    allowed_tokens: Iterable[int],
    *,
    disallowed_value: float = -math.inf,
) -> list[float]:
    """Mask logits to an automaton state's allowed token ids.

    This function performs the local support intersection only.  A caller must
    run softmax after masking; the resulting conditional distribution is a
    distribution-changing operation unless the application explicitly defines
    the target as the grammar-conditioned model.
    """
    values = _as_floats(logits, name="logits")
    allowed = {int(i) for i in allowed_tokens}
    if not allowed:
        raise ValueError("grammar state has an empty allowed-token set")
    if min(allowed) < 0 or max(allowed) >= len(values):
        raise ValueError("allowed token id is outside the vocabulary")
    if math.isnan(disallowed_value):
        raise ValueError("disallowed_value must not be NaN")
    return [value if index in allowed else float(disallowed_value) for index, value in enumerate(values)]


grammar_mask = grammar_filter


def acceptance_probability(target_probability: float, draft_probability: float) -> float:
    """Speculative sampling acceptance α=min(1,p/q) for one token."""
    p = float(target_probability)
    q = float(draft_probability)
    if not (math.isfinite(p) and math.isfinite(q)) or p < 0.0 or q < 0.0:
        raise ValueError("p and q must be finite and non-negative")
    if q == 0.0:
        return 1.0 if p > 0.0 else 0.0
    return min(1.0, p / q)


def residual_distribution(target: Sequence[float], draft: Sequence[float]) -> list[float]:
    """Compute [p-q]+/sum([p-q]+), with p as fallback if residual mass is zero."""
    p = normalize(target)
    q = normalize(draft)
    if len(p) != len(q):
        raise ValueError("target and draft must have the same vocabulary size")
    positive = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    mass = math.fsum(positive)
    return normalize(positive) if mass > 0.0 else p


def speculative_acceptance_simulation(
    target: Sequence[float],
    draft: Sequence[float],
    draft_tokens: Sequence[int],
    *,
    seed: int = 0,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Run one deterministic draft/verify round and return an audit record.

    ``target`` and ``draft`` are single-step distributions.  For a multi-token
    draft, the same vectors are used for each position as a deliberately small
    simulation; a production implementation supplies a conditional p_i and q_i
    for every position.  On rejection, the residual correction is sampled and
    the suffix is discarded.  A draft accepted at every position is returned as
    is; a real decoder would request one additional target token.
    """
    p = normalize(target)
    q = normalize(draft)
    if len(p) != len(q):
        raise ValueError("target and draft must have the same vocabulary size")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    generator = rng if rng is not None else random.Random(seed)
    output: list[int] = []
    probabilities: list[float] = []
    accepted = 0
    for raw_token in draft_tokens:
        token = int(raw_token)
        if token < 0 or token >= len(p):
            raise ValueError("draft token is outside the vocabulary")
        alpha = acceptance_probability(p[token], q[token])
        probabilities.append(alpha)
        # A draw of exactly zero must still reject when alpha=0.  Using <= here
        # silently accepted zero-probability tokens under a forced RNG.
        if generator.random() < alpha:
            output.append(token)
            accepted += 1
            continue
        corrected = residual_distribution(p, q)
        output.append(sample_categorical(corrected, rng=generator))
        break
    return {
        "accepted": accepted,
        "draft_length": len(draft_tokens),
        "acceptance_rate": accepted / len(draft_tokens) if draft_tokens else 0.0,
        "acceptance_probabilities": probabilities,
        "output_tokens": output,
        "rejected": accepted < len(draft_tokens),
        "seed": seed,
    }


# Names kept explicit for readers searching for the operation.
simulate_speculative_acceptance = speculative_acceptance_simulation


def sample_categorical(probabilities: Sequence[float], *, rng: random.Random | None = None) -> int:
    """Sample an index with a supplied RNG; deterministic for a fixed seed."""
    values = normalize(probabilities)
    generator = rng if rng is not None else random.Random(0)
    draw = generator.random()
    cumulative = 0.0
    for index, probability in enumerate(values):
        cumulative += probability
        if draw < cumulative or index == len(values) - 1:
            return index
    raise AssertionError("unreachable")


def distinct_n(sequences: Sequence[Sequence[int]], n: int = 2) -> float:
    """Fraction of unique n-grams among all generated n-grams."""
    if n <= 0:
        raise ValueError("n must be positive")
    grams: list[tuple[int, ...]] = []
    for sequence in sequences:
        grams.extend(tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1))
    return len(set(grams)) / len(grams) if grams else 0.0


def _toy_experiment(seed: int = 7) -> dict[str, Any]:
    logits = [2.0, 1.0, 0.2, -0.8, -1.5]
    original = logits_to_probabilities(logits)
    rng = random.Random(seed)
    samples: list[list[int]] = []
    for _ in range(64):
        sequence = [sample_categorical(top_p(original, 0.9), rng=rng) for _ in range(8)]
        samples.append(sequence)
    draft = logits_to_probabilities([1.8, 1.1, 0.1, -0.7, -1.3])
    draft_tokens = [sample_categorical(draft, rng=rng) for _ in range(4)]
    return {
        "seed": seed,
        "probabilities": original,
        "entropy_bits": entropy(original, base=2.0),
        "top_p_support": sum(p > 0.0 for p in top_p(original, 0.9)),
        "distinct_2": distinct_n(samples, n=2),
        "speculative_round": speculative_acceptance_simulation(original, draft, draft_tokens, seed=seed),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile for a small deterministic experiment."""
    numbers = sorted(float(value) for value in values)
    if not numbers or not 0.0 <= quantile <= 1.0:
        raise ValueError("values must be non-empty and quantile must be in [0, 1]")
    rank = max(0, min(len(numbers) - 1, math.ceil(quantile * len(numbers)) - 1))
    return numbers[rank]


def multi_seed_experiment(seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    """Aggregate the toy experiment without pretending to measure model latency.

    The summary reports per-seed min/median/p95/max for diversity and acceptance.
    These percentiles describe the five deterministic repetitions, not serving
    request latency.  A caller can pass any non-empty finite list of integer
    seeds and receives the same JSON for the same list.
    """
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must be a non-empty sequence of integers")
    records = [_toy_experiment(seed) for seed in seeds]
    def stats(key: str) -> dict[str, float]:
        values = [float(record[key]) for record in records]
        return {
            "min": min(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    acceptance = [float(record["speculative_round"]["acceptance_rate"]) for record in records]
    accepted_lengths = [float(record["speculative_round"]["accepted"]) for record in records]
    return {
        "seeds": list(seeds),
        "per_seed": records,
        "summary": {
            "distinct_2": stats("distinct_2"),
            "acceptance_rate": {
                "min": min(acceptance),
                "p50": percentile(acceptance, 0.5),
                "p95": percentile(acceptance, 0.95),
                "max": max(acceptance),
            },
            "accepted_tokens": {
                "min": min(accepted_lengths),
                "p50": percentile(accepted_lengths, 0.5),
                "p95": percentile(accepted_lengths, 0.95),
                "max": max(accepted_lengths),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--aggregate", action="store_true", help="run deterministic seeds 0..4 and summarize toy metrics")
    args = parser.parse_args()
    result = multi_seed_experiment() if args.aggregate else _toy_experiment(args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
