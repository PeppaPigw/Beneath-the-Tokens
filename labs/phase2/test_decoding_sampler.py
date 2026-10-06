from __future__ import annotations

import math
import random
import unittest

from decoding_sampler import (
    acceptance_probability,
    apply_temperature,
    distinct_n,
    entropy,
    grammar_filter,
    logits_to_probabilities,
    min_p,
    mirostat_update,
    multi_seed_experiment,
    residual_distribution,
    sample_categorical,
    speculative_acceptance_simulation,
    top_k,
    top_p,
    typical,
    grammar_mask,
    min_p_filter,
    mirostat_step,
    temperature_scale,
    top_k_filter,
    top_p_filter,
    typical_filter,
)


class DecodingSamplerTests(unittest.TestCase):
    def test_discoverable_aliases_match_core_functions(self) -> None:
        self.assertEqual(temperature_scale([2.0, 0.0], 2.0), apply_temperature([2.0, 0.0], 2.0))
        p = [0.5, 0.3, 0.2]
        self.assertEqual(top_k_filter(p, 2), top_k(p, 2))
        self.assertEqual(top_p_filter(p, 0.8), top_p(p, 0.8))
        self.assertEqual(typical_filter(p, 0.8), typical(p, 0.8))
        self.assertEqual(min_p_filter(p, 0.2), min_p(p, 0.2))
        self.assertEqual(mirostat_step(1.0, 1.2, 4.0), mirostat_update(1.0, 1.2, 4.0))
        self.assertEqual(grammar_mask([1.0, 2.0], [1]), grammar_filter([1.0, 2.0], [1]))

    def test_softmax_is_stable_and_normalized(self) -> None:
        probs = logits_to_probabilities([1000.0, 999.0, 998.0])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[0], probs[1])

    def test_temperature_changes_entropy_in_expected_direction(self) -> None:
        logits = [3.0, 1.0, -1.0]
        cold = logits_to_probabilities(logits, temperature=0.5)
        hot = logits_to_probabilities(logits, temperature=2.0)
        self.assertLess(entropy(cold), entropy(hot))
        self.assertEqual(apply_temperature([2.0, 0.0], 2.0), [1.0, 0.0])

    def test_truncation_supports_and_normalizes(self) -> None:
        p = [0.5, 0.25, 0.15, 0.1]
        self.assertEqual(sum(x > 0.0 for x in top_k(p, 2)), 2)
        self.assertEqual(sum(x > 0.0 for x in top_p(p, 0.7)), 2)
        self.assertAlmostEqual(sum(top_p(p, 0.7)), 1.0)
        self.assertAlmostEqual(sum(typical(p, 0.9)), 1.0)
        self.assertEqual(sum(x > 0.0 for x in min_p(p, 0.5)), 2)

    def test_invalid_parameters_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            logits_to_probabilities([1.0], temperature=0.0)
        with self.assertRaises(ValueError):
            top_k([1.0, 0.0], 0)
        with self.assertRaises(ValueError):
            top_p([1.0], 0.0)
        with self.assertRaises(ValueError):
            grammar_filter([0.0, 1.0], [])
        with self.assertRaises(ValueError):
            grammar_filter([0.0, 1.0], [2])

    def test_mirostat_update_moves_against_surprisal_error(self) -> None:
        target_mu = math.log(4.0)
        self.assertAlmostEqual(mirostat_update(target_mu, target_mu + 1.0, 4.0, eta=0.2), target_mu - 0.2)
        self.assertAlmostEqual(mirostat_update(target_mu, target_mu - 1.0, 4.0, eta=0.2), target_mu + 0.2)

    def test_grammar_filter_masks_without_mutating(self) -> None:
        logits = [2.0, 3.0, 4.0]
        masked = grammar_filter(logits, [0, 2])
        self.assertEqual(logits, [2.0, 3.0, 4.0])
        self.assertEqual(masked[0], 2.0)
        self.assertTrue(math.isinf(masked[1]) and masked[1] < 0)
        self.assertEqual(masked[2], 4.0)
        self.assertAlmostEqual(sum(logits_to_probabilities(masked)), 1.0)

    def test_residual_is_model_correction(self) -> None:
        p = [0.6, 0.3, 0.1]
        q = [0.3, 0.6, 0.1]
        residual = residual_distribution(p, q)
        self.assertAlmostEqual(sum(residual), 1.0)
        self.assertEqual(residual, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(acceptance_probability(0.6, 0.3), 1.0)
        self.assertAlmostEqual(acceptance_probability(0.3, 0.6), 0.5)

    def test_speculative_simulation_is_reproducible_and_stops_after_reject(self) -> None:
        p = [0.7, 0.2, 0.1]
        q = [0.4, 0.5, 0.1]
        draft = [1, 0, 2, 1]
        first = speculative_acceptance_simulation(p, q, draft, seed=11)
        second = speculative_acceptance_simulation(p, q, draft, seed=11)
        self.assertEqual(first, second)
        self.assertLessEqual(first["accepted"], len(draft))
        self.assertLessEqual(len(first["output_tokens"]), len(draft))

    def test_zero_draw_does_not_accept_zero_probability_token(self) -> None:
        class ZeroRng:
            def random(self) -> float:
                return 0.0

        # q proposes token 0, while p(0)=0, so alpha=0. Strict '<' must reject
        # even when the deterministic draw is exactly zero.
        result = speculative_acceptance_simulation([0.0, 1.0], [1.0, 0.0], [0], rng=ZeroRng())
        self.assertEqual(result["accepted"], 0)
        self.assertTrue(result["rejected"])

    def test_categorical_sampling_reproducible(self) -> None:
        rng1, rng2 = random.Random(4), random.Random(4)
        self.assertEqual([sample_categorical([0.2, 0.8], rng=rng1) for _ in range(20)], [sample_categorical([0.2, 0.8], rng=rng2) for _ in range(20)])

    def test_distinct_n_bounds(self) -> None:
        value = distinct_n([[0, 1, 0], [0, 1, 0]], 2)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_multi_seed_summary_is_deterministic_and_has_percentiles(self) -> None:
        first = multi_seed_experiment((0, 1, 2, 3, 4))
        second = multi_seed_experiment((0, 1, 2, 3, 4))
        self.assertEqual(first, second)
        self.assertEqual(first["seeds"], [0, 1, 2, 3, 4])
        for metric in ("distinct_2", "acceptance_rate", "accepted_tokens"):
            self.assertIn("p50", first["summary"][metric])
            self.assertLessEqual(first["summary"][metric]["min"], first["summary"][metric]["max"])


if __name__ == "__main__":
    unittest.main()
