from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy.linalg import solve_triangular

from ..config import MemoryConfig
from ..schemas import SessionEvent
from ..memory_utils import (
    build_chunks,
    chunk_record_dedupe_key,
    dedupe_chunk_records,
    render_chunk_records,
)

def embedding_to_feature_saturation_features(embeddings):
    """
    Convert one or more dense embeddings into a fixed non-negative feature map.

    For an embedding e in R^d:

        z(e) = [max(e, 0), max(-e, 0)] / ||e||_1

    Hence z(e) is non-negative, has dimension 2d, and has total mass 1.
    The mapping itself is fixed for the entire stream; it does not depend on
    previously seen or future chunks.
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    single = (arr.ndim == 1)

    if single:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(
            "coverage feature transform expects shape (d,) or (n, d); "
            f"got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("coverage feature transform received NaN/inf.")

    positive = np.maximum(arr, 0.0)
    negative = np.maximum(-arr, 0.0)
    features = np.concatenate([positive, negative], axis=1)

    mass = features.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("coverage feature transform received a zero embedding.")

    features = features / mass

    if single:
        return features[0]
    return features

def feature_saturation_coverage_value(feature_mass, tau):
    """
    Fixed normalized monotone-submodular feature coverage:

        C(S) = sum_j log(1 + tau * m_j(S)) / tau,
        m_j(S) = sum_{u in S} z_j(u).

    For tau > 0, log(1 + tau*x)/tau is increasing and concave, so composing
    it with non-negative modular feature mass yields a monotone submodular
    set function.  C(empty)=0.
    """
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("FEATURE_COVERAGE_TAU must be finite and > 0.")

    mass = np.asarray(feature_mass, dtype=np.float64)
    if mass.ndim != 1:
        raise ValueError(f"feature_mass must be 1D; got {mass.shape}")
    if not np.all(np.isfinite(mass)):
        raise ValueError("feature_mass contains NaN/inf.")

    # Tiny negative values can arise only from floating-point subtraction in
    # leave-one-out diagnostics; they are not semantically negative mass.
    mass = np.maximum(mass, 0.0)

    return float(np.log1p(tau * mass).sum() / tau)

class StreamingCovDivState:
    """Faithful incremental implementation of PDF Algorithms 4-5.

    The maintained objective is query-independent and FIXED during Stage-1:

        f(S) = lambda_coverage * feature_saturation_coverage(S)
             + lambda_diversity * logdet(I + K_S)

    Coverage uses a fixed non-negative transformation of each E5 embedding:
    L1-normalized positive/negative coordinate splitting, followed by a
    concave log1p saturation over accumulated feature mass.  It therefore
    requires neither hand-written semantic anchors nor future stream elements.

    Algorithm 4 PrefixOneStream(hbar=2) supplies the anytime calibration H_t.
    Algorithm 5 maintains the calibrated geometric value guesses. Candidate
    states use lineage-based copy-on-write and DPP Cholesky marginal/append
    reuse.

    The density coefficient is computed internally from epsilon/beta:
    alpha=(1-eta)/(1+q_beta), q_beta=max(1/2,1-beta).
    """

    _GRID_TOL = 1e-12
    _DPP_INITIAL_CAPACITY = 16

    def __init__(
        self,
        compressor,
        token_counter,
        token_budget,
        lambda_coverage=0.2,
        lambda_diversity=0.8,
        coverage_tau=1.0,
        epsilon=0.3,
        beta=1.0 / 8.0,
        strict_beta_cap=False,
    ):
        self.compressor = compressor
        self.token_counter = token_counter
        self.lambda_coverage = float(lambda_coverage)
        self.lambda_diversity = float(lambda_diversity)
        if self.lambda_coverage < 0.0 or self.lambda_diversity < 0.0:
            raise ValueError("streaming objective weights must be non-negative.")
        self.token_budget = int(token_budget)
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive.")

        # One-parameter Corollary-2 choice: xi = eta = epsilon.
        self.epsilon = float(epsilon)
        self.xi = self.epsilon
        self.eta = self.epsilon
        self.beta = float(beta)
        self.legacy_alpha = None
        self.h = 2
        self.passes = 1
        self.strict_beta_cap = bool(strict_beta_cap)

        if not (0.0 < self.xi < 1.0 and 0.0 < self.eta < 1.0):
            raise ValueError("CalibratedGranularStream requires xi, eta in (0, 1).")
        if not (0.0 < self.beta <= 1.0):
            raise ValueError("beta is the ex-ante granularity cap and must be in (0, 1].")
        if self.passes != 1:
            raise ValueError("CalibratedGranularStream is one-pass; passes must be 1.")
        if self.h != 2:
            raise ValueError("PDF Algorithm 4 uses PrefixOneStream with hbar=2.")

        self.prefix_one_stream_hbar = 2
        self.onestream_C0 = 16.0
        self.q_beta = max(0.5, 1.0 - self.beta)
        self.density_alpha = (1.0 - self.eta) / (1.0 + self.q_beta)
        self.gamma = self.density_alpha * self.q_beta

        self.coverage_tau = float(coverage_tau)
        if not np.isfinite(self.coverage_tau) or self.coverage_tau <= 0.0:
            raise ValueError("coverage_tau must be finite and > 0.")

        # The feature rule is fixed from the start.  We cache the embedding
        # dimension when available, but can also learn it from the first
        # arriving embedding for compatibility with lightweight test doubles.
        self.embedding_dim = None
        semantic_model = getattr(self.compressor, "semantic_model", None)
        get_dim = getattr(semantic_model, "get_sentence_embedding_dimension", None)
        if callable(get_dim):
            model_dim = get_dim()
            if model_dim is not None:
                model_dim = int(model_dim)
                if model_dim > 0:
                    self.embedding_dim = model_dim

        self.coverage_feature_dim = (
            2 * self.embedding_dim
            if self.embedding_dim is not None
            else None
        )

        self.best_sync_rtol = 1e-10
        self.best_sync_atol = 1e-10

        # Retained-element store.  Unlike the old implementation, discarded raw
        # elements are garbage-collected after each arrival.  Only elements
        # referenced by Algorithm-4/5 states are retained.
        self.embedding_store: Dict[int, np.ndarray] = {}
        self.record_store: Dict[int, Dict[str, Any]] = {}
        self.next_stream_index = 0
        self.total_feasible_seen = 0
        self.retained_elements_peak = 0
        self.gc_deleted_elements_total = 0

        # O(1) deterministic lineage IDs for COW density states.
        self._next_signature_id = 1
        self._signature_transition_ids = {}

        self._empty_candidate_state = self._new_empty_state()
        self.candidates: Dict[float, Dict[str, Any]] = {}

        # Algorithm 4 PrefixOneStream state.
        self.calibration_blocks: List[List[int]] = [[]]
        self.calibration_block_costs: List[int] = [0]
        self.calibration_W_state = self._new_empty_state()
        self.best_singleton_state = self._new_empty_state()
        self.best_singleton_value = 0.0
        self.best_calibration_state = self._new_empty_state()
        self.H = 0.0

        # Algorithm 5 Q and LB.
        self.best_state = self._new_empty_state()
        self.best_value = 0.0
        self.best_source = "empty"

        # Diagnostics.
        self.thresholds_created = 0
        self.thresholds_removed_total = 0
        self.thresholds_pruned_before_processing = 0
        self.thresholds_pruned_after_processing = 0
        self.thresholds_peak = 0

        self.accepted_elements = 0
        self.blocked_high_density_events = 0
        self.dropped_oversized_chunks = 0
        self.granularity_cap_violations = 0
        self.processed_chunks = 0
        self.feasible_chunks = 0

        self.singleton_oracle_calls = 0
        self.calibration_marginal_oracle_calls = 0
        self.calibration_certificate_oracle_calls = 0
        self.marginal_oracle_calls = 0
        self.oracle_calls_approx = 0
        self.numerical_oracle_failures = 0

        self.compacted_to_best = False
        self.seen_chunks_before_compaction = 0
        self.deleted_unretained_chunks = 0
        self.deleted_unretained_embeddings = 0
        self.active_guesses_before_compaction = 0
        self.active_element_copies_before_compaction = 0
        self.active_element_copies_peak = 0
        self.element_copies_peak_including_best = 0

        # Optimization diagnostics.
        self.sim_to_retained_batch_calls = 0
        self.sim_to_seen_full_vector_calls = 0  # compatibility alias/counter

        self.state_signature_cache_hits = 0
        self.state_signature_cache_misses = 0
        self.state_signature_unique_states_total = 0

        self.cow_parent_groups_processed = 0
        self.cow_child_states_created = 0
        self.cow_inplace_group_updates = 0
        self.cow_shared_child_assignments = 0
        self.cow_state_copies_avoided = 0
        self.unique_candidate_state_peak = 0

        self.dpp_solve_calls = 0
        self.dpp_solve_reused_for_append = 0
        self.dpp_buffer_expansions = 0
        self.dpp_buffer_allocated_cells_peak = 0
        self.best_state_dpp_matrix_copies = 0

        self._assert_best_sync()
        self._update_memory_peaks()

    # ==================================================================
    # Generic state helpers
    # ==================================================================

    def _new_empty_state(self):
        return {
            "selected": set(),
            "selected_order": [],
            "current_cost": 0,
            "signature_id": 0,
            "coverage_mass": (
                np.zeros(self.coverage_feature_dim, dtype=np.float64)
                if self.coverage_feature_dim is not None
                else None
            ),
            "dpp_buffer": None,
            "dpp_size": 0,
            "dpp_capacity": 0,
            "coverage_value": 0.0,
            "diversity_value": 0.0,
            "value": 0.0,
        }

    def _state_value(self, state):
        return float(state.get("value", 0.0))

    def _copy_state(self, state, include_dpp_buffer=True):
        dpp_buffer = state.get("dpp_buffer")
        if include_dpp_buffer and dpp_buffer is not None:
            dpp_buffer = dpp_buffer.copy()
        else:
            dpp_buffer = None

        return {
            "selected": set(state["selected"]),
            "selected_order": list(state["selected_order"]),
            "current_cost": int(state["current_cost"]),
            "signature_id": int(state.get("signature_id", 0)),
            "coverage_mass": (
                None
                if state.get("coverage_mass") is None
                else np.asarray(
                    state["coverage_mass"], dtype=np.float64
                ).copy()
            ),
            "dpp_buffer": dpp_buffer,
            "dpp_size": int(state.get("dpp_size", 0)) if dpp_buffer is not None else 0,
            "dpp_capacity": (
                int(state.get("dpp_capacity", 0)) if dpp_buffer is not None else 0
            ),
            "coverage_value": float(state.get("coverage_value", 0.0)),
            "diversity_value": float(state.get("diversity_value", 0.0)),
            "value": float(state.get("value", 0.0)),
        }

    def _snapshot_state(self, state):
        # Algorithm-5 Q / Algorithm-4 best certificates are output snapshots;
        # they never need another marginal, so no DPP matrix is copied.
        out = self._copy_state(state, include_dpp_buffer=False)
        self.best_state_dpp_matrix_copies += 0
        return out

    def _assert_best_sync(self):
        cached = self._state_value(self.best_state)
        if not np.isclose(
            cached,
            self.best_value,
            rtol=self.best_sync_rtol,
            atol=self.best_sync_atol,
        ):
            raise RuntimeError(
                f"best_state/value desynchronized: state={cached}, best={self.best_value}"
            )

    def _set_best_state(self, state, source):
        value = self._state_value(state)
        if value + self.best_sync_atol < self.best_value:
            return
        self.best_state = self._snapshot_state(state)
        self.best_value = value
        self.best_source = str(source)
        self._assert_best_sync()

    def _state_signature(self, state):
        return int(state.get("signature_id", 0))

    def _child_signature(self, parent_signature, new_idx):
        key = (int(parent_signature), int(new_idx))
        child = self._signature_transition_ids.get(key)
        if child is None:
            child = self._next_signature_id
            self._next_signature_id += 1
            self._signature_transition_ids[key] = child
        return child

    def _active_element_copies(self):
        return sum(
            len(state["selected_order"])
            for state in self.candidates.values()
        )

    def _update_memory_peaks(self):
        self.active_element_copies_peak = max(
            self.active_element_copies_peak,
            self._active_element_copies(),
        )
        self.element_copies_peak_including_best = max(
            self.element_copies_peak_including_best,
            self._active_element_copies()
            + len(self.best_state["selected_order"])
            + sum(len(b) for b in self.calibration_blocks)
            + len(self.best_calibration_state["selected_order"])
            + len(self.best_singleton_state["selected_order"]),
        )
        unique_states = len({
            id(state) for state in self.candidates.values()
        })
        self.unique_candidate_state_peak = max(
            self.unique_candidate_state_peak, unique_states
        )
        self.retained_elements_peak = max(
            self.retained_elements_peak, len(self.embedding_store)
        )

    # ==================================================================
    # Retained embedding/record store
    # ==================================================================

    def _normalize_embedding(self, emb):
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got {emb.shape}")
        if self.embedding_dim is None:
            self.embedding_dim = int(emb.shape[0])
            self.coverage_feature_dim = 2 * self.embedding_dim
        elif emb.shape[0] != self.embedding_dim:
            raise ValueError(
                f"stream dim={emb.shape[0]}, expected dim={self.embedding_dim}"
            )
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("incoming embedding has invalid norm")
        return (emb / norm).astype(np.float32, copy=False)

    def _sim_new_to_retained(self, new_emb):
        if not self.embedding_store:
            return {}
        ids = sorted(self.embedding_store)
        E = np.stack([self.embedding_store[i] for i in ids], axis=0)
        sims = (E @ new_emb).astype(np.float64)
        self.sim_to_retained_batch_calls += 1
        self.sim_to_seen_full_vector_calls += 1
        return {idx: float(sim) for idx, sim in zip(ids, sims)}

    def _live_element_ids(self):
        live = set(self.best_state["selected_order"])
        live.update(self.best_calibration_state["selected_order"])
        live.update(self.best_singleton_state["selected_order"])
        for block in self.calibration_blocks:
            live.update(block)
        for state in self.candidates.values():
            live.update(state["selected_order"])
        return live

    def _gc_unreferenced_elements(self):
        live = self._live_element_ids()
        dead = [idx for idx in self.embedding_store if idx not in live]
        for idx in dead:
            self.embedding_store.pop(idx, None)
            self.record_store.pop(idx, None)
        self.gc_deleted_elements_total += len(dead)
        return len(dead)

    # ==================================================================
    # Objective helpers
    # ==================================================================

    def _embedding_to_coverage_features(self, emb):
        features = embedding_to_feature_saturation_features(emb)
        features = np.asarray(features, dtype=np.float64)

        if features.ndim != 1:
            raise ValueError(
                f"coverage feature vector must be 1D; got {features.shape}"
            )

        if self.coverage_feature_dim is None:
            self.coverage_feature_dim = int(features.shape[0])
        elif features.shape[0] != self.coverage_feature_dim:
            raise ValueError(
                "coverage feature dimension mismatch: "
                f"{features.shape[0]} != {self.coverage_feature_dim}"
            )

        return features

    def _coverage_value_from_mass(self, feature_mass):
        if feature_mass is None:
            return 0.0
        return feature_saturation_coverage_value(
            feature_mass,
            self.coverage_tau,
        )

    def _ensure_dpp_capacity(self, state, needed):
        needed = int(needed)
        if needed <= int(state.get("dpp_capacity", 0)):
            return
        old = state.get("dpp_buffer")
        old_size = int(state.get("dpp_size", 0))
        capacity = max(
            self._DPP_INITIAL_CAPACITY,
            int(state.get("dpp_capacity", 0)) or self._DPP_INITIAL_CAPACITY,
        )
        while capacity < needed:
            capacity *= 2
        new = np.zeros((capacity, capacity), dtype=np.float64)
        if old is not None and old_size > 0:
            new[:old_size, :old_size] = old[:old_size, :old_size]
        state["dpp_buffer"] = new
        state["dpp_capacity"] = capacity
        self.dpp_buffer_expansions += 1
        self.dpp_buffer_allocated_cells_peak = max(
            self.dpp_buffer_allocated_cells_peak, capacity * capacity
        )

    def _dpp_view(self, state):
        size = int(state.get("dpp_size", 0))
        buf = state.get("dpp_buffer")
        if size <= 0 or buf is None:
            return None
        return buf[:size, :size]

    def _dpp_marginal_and_update_new(self, state, sim_lookup):
        if self.lambda_diversity <= 0.0:
            return 0.0, None

        selected_order = state["selected_order"]
        m = len(selected_order)
        k_cc = 2.0  # diag(I + K), normalized embedding K_cc=1.

        if m == 0:
            return float(np.log(k_cc)), {
                "z": np.zeros(0, dtype=np.float64),
                "schur": k_cc,
            }

        L = self._dpp_view(state)
        if L is None or int(state["dpp_size"]) != m:
            self.numerical_oracle_failures += 1
            return -1e18, None

        try:
            v = np.asarray(
                [sim_lookup[idx] for idx in selected_order],
                dtype=np.float64,
            ).reshape(-1, 1)
        except KeyError:
            self.numerical_oracle_failures += 1
            return -1e18, None

        try:
            z = solve_triangular(L, v, lower=True, check_finite=False)
            self.dpp_solve_calls += 1
        except Exception:
            self.numerical_oracle_failures += 1
            return -1e18, None

        if not np.all(np.isfinite(z)):
            self.numerical_oracle_failures += 1
            return -1e18, None

        schur = k_cc - float((z.T @ z)[0, 0])
        if (
            not np.isfinite(schur)
            or schur <= self.compressor.dpp_schur_eps
        ):
            self.numerical_oracle_failures += 1
            return -1e18, None

        return float(np.log(schur)), {
            "z": z.ravel(),
            "schur": float(schur),
        }

    def _append_dpp_from_update(self, state, dpp_update):
        if self.lambda_diversity <= 0.0:
            return True
        if dpp_update is None:
            return False

        m = len(state["selected_order"])
        if int(state["dpp_size"]) != m:
            return False

        z = np.asarray(dpp_update["z"], dtype=np.float64).reshape(-1)
        schur = float(dpp_update["schur"])
        if (
            z.shape[0] != m
            or not np.isfinite(schur)
            or schur <= self.compressor.dpp_schur_eps
        ):
            return False

        self._ensure_dpp_capacity(state, m + 1)
        buf = state["dpp_buffer"]
        if m > 0:
            buf[m, :m] = z
        buf[m, m] = np.sqrt(schur)
        state["dpp_size"] = m + 1
        self.dpp_solve_reused_for_append += 1
        return True

    def _marginal_gain_new_parts(
        self,
        state,
        new_idx,
        coverage_features,
        sim_lookup,
    ):
        if new_idx in state["selected"]:
            return -1e18, 0.0, 0.0, None, None

        old_mass = state.get("coverage_mass")
        if old_mass is None:
            old_mass = np.zeros_like(coverage_features, dtype=np.float64)
        else:
            old_mass = np.asarray(old_mass, dtype=np.float64)

        if old_mass.shape != coverage_features.shape:
            raise RuntimeError(
                "coverage state/feature shape mismatch: "
                f"{old_mass.shape} != {coverage_features.shape}"
            )

        new_coverage_mass = old_mass + coverage_features
        old_coverage_value = float(state.get("coverage_value", 0.0))
        new_coverage_value = self._coverage_value_from_mass(
            new_coverage_mass
        )
        coverage_gain = max(
            float(new_coverage_value - old_coverage_value),
            0.0,
        )

        diversity_gain, dpp_update = self._dpp_marginal_and_update_new(
            state, sim_lookup
        )
        if diversity_gain <= -1e17:
            return -1e18, coverage_gain, diversity_gain, new_coverage_mass, None

        total_gain = float(
            self.lambda_coverage * coverage_gain
            + self.lambda_diversity * diversity_gain
        )
        return (
            total_gain,
            coverage_gain,
            diversity_gain,
            new_coverage_mass,
            dpp_update,
        )

    def _append_new_to_state(
        self,
        state,
        new_idx,
        coverage_features,
        sim_lookup,
        cost,
        coverage_gain=None,
        diversity_gain=None,
        new_coverage_mass=None,
        dpp_update=None,
    ):
        if new_idx in state["selected"]:
            return False

        if (
            coverage_gain is None
            or diversity_gain is None
            or new_coverage_mass is None
        ):
            (
                total_gain,
                coverage_gain,
                diversity_gain,
                new_coverage_mass,
                dpp_update,
            ) = self._marginal_gain_new_parts(
                state, new_idx, coverage_features, sim_lookup
            )
            if not np.isfinite(total_gain):
                return False

        old_signature = self._state_signature(state)
        if not self._append_dpp_from_update(state, dpp_update):
            return False

        state["selected"].add(new_idx)
        state["selected_order"].append(new_idx)
        state["current_cost"] += int(cost)
        state["signature_id"] = self._child_signature(old_signature, new_idx)
        state["coverage_mass"] = np.asarray(
            new_coverage_mass, dtype=np.float64
        ).copy()
        state["coverage_value"] = float(
            state.get("coverage_value", 0.0) + coverage_gain
        )
        state["diversity_value"] = float(
            state.get("diversity_value", 0.0) + diversity_gain
        )
        state["value"] = float(
            self.lambda_coverage * state["coverage_value"]
            + self.lambda_diversity * state["diversity_value"]
        )
        return True

    def _singleton_state_for_new(self, new_idx, coverage_features, cost):
        if cost > self.token_budget:
            return None
        state = self._new_empty_state()
        empty_lookup = {}
        (
            gain,
            coverage_gain,
            diversity_gain,
            new_coverage_mass,
            dpp_update,
        ) = self._marginal_gain_new_parts(
            state, new_idx, coverage_features, empty_lookup
        )
        if not np.isfinite(gain):
            return None
        ok = self._append_new_to_state(
            state,
            new_idx,
            coverage_features,
            empty_lookup,
            cost,
            coverage_gain=coverage_gain,
            diversity_gain=diversity_gain,
            new_coverage_mass=new_coverage_mass,
            dpp_update=dpp_update,
        )
        return state if ok else None

    def _rebuild_state_from_ids(self, ids):
        state = self._new_empty_state()
        for idx in ids:
            emb = self.embedding_store.get(idx)
            record = self.record_store.get(idx)
            if emb is None or record is None:
                raise RuntimeError(f"retained element {idx} missing during state rebuild")
            coverage_features = self._embedding_to_coverage_features(emb)
            sim_lookup = {
                old: float(self.embedding_store[old] @ emb)
                for old in state["selected_order"]
            }
            (
                gain,
                coverage_gain,
                diversity_gain,
                new_coverage_mass,
                dpp_update,
            ) = self._marginal_gain_new_parts(
                state, idx, coverage_features, sim_lookup
            )
            if not np.isfinite(gain):
                raise RuntimeError("failed to rebuild retained DPP state")
            if not self._append_new_to_state(
                state,
                idx,
                coverage_features,
                sim_lookup,
                int(record["stream_token_cost"]),
                coverage_gain=coverage_gain,
                diversity_gain=diversity_gain,
                new_coverage_mass=new_coverage_mass,
                dpp_update=dpp_update,
            ):
                raise RuntimeError("failed to append during retained state rebuild")
        return state

    def _snapshot_for_ids(self, ids):
        ids = list(ids)
        if not ids:
            return self._new_empty_state()

        rows = [self.embedding_store[i] for i in ids]
        E = np.stack(rows, axis=0).astype(np.float64, copy=False)

        # Exact fixed feature-saturation coverage value.
        coverage_features = embedding_to_feature_saturation_features(E)
        coverage_mass = np.asarray(
            coverage_features, dtype=np.float64
        ).sum(axis=0)
        coverage_value = self._coverage_value_from_mass(
            coverage_mass
        )

        # Exact logdet(I + K_S).
        K = E @ E.T
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)
        sign, logdet = np.linalg.slogdet(
            np.eye(len(ids), dtype=np.float64) + K
        )
        if sign <= 0 or not np.isfinite(logdet):
            raise RuntimeError("non-positive DPP determinant in certificate snapshot")

        diversity_value = float(logdet)
        cost = sum(
            int(self.record_store[i]["stream_token_cost"])
            for i in ids
        )
        value = float(
            self.lambda_coverage * coverage_value
            + self.lambda_diversity * diversity_value
        )
        return {
            "selected": set(ids),
            "selected_order": ids,
            "current_cost": int(cost),
            "signature_id": 0,
            "coverage_mass": coverage_mass,
            "dpp_buffer": None,
            "dpp_size": 0,
            "dpp_capacity": 0,
            "coverage_value": coverage_value,
            "diversity_value": diversity_value,
            "value": value,
        }

    # ==================================================================
    # Algorithm 4: PrefixOneStream(hbar=2)
    # ==================================================================

    def _calibration_W_order(self):
        return [idx for block in self.calibration_blocks for idx in block]

    def _tail_b_order(self, order):
        total = 0
        out_rev = []
        for idx in reversed(order):
            cost = int(self.record_store[idx]["stream_token_cost"])
            if total + cost > self.token_budget:
                break
            out_rev.append(idx)
            total += cost
        return list(reversed(out_rev))

    def _prefix_onestream_update(
        self,
        new_idx,
        cost,
        coverage_features,
        sim_lookup,
        singleton_state,
        singleton_value,
    ):
        # Algorithm 4 lines 4-6: compare u against retained union W.
        f_W = self._state_value(self.calibration_W_state)
        (
            gain_W,
            coverage_gain,
            diversity_gain,
            new_coverage_mass,
            dpp_update,
        ) = self._marginal_gain_new_parts(
            self.calibration_W_state,
            new_idx,
            coverage_features,
            sim_lookup,
        )
        self.calibration_marginal_oracle_calls += 1
        self.oracle_calls_approx += 1

        if (
            np.isfinite(gain_W)
            and gain_W / float(cost) >= f_W / float(self.token_budget)
        ):
            if not self._append_new_to_state(
                self.calibration_W_state,
                new_idx,
                coverage_features,
                sim_lookup,
                cost,
                coverage_gain=coverage_gain,
                diversity_gain=diversity_gain,
                new_coverage_mass=new_coverage_mass,
                dpp_update=dpp_update,
            ):
                raise RuntimeError("PrefixOneStream failed to append accepted element")

            self.calibration_blocks[-1].append(new_idx)
            self.calibration_block_costs[-1] += int(cost)

            # Algorithm 4 lines 7-10.
            if self.calibration_block_costs[-1] >= self.token_budget:
                if len(self.calibration_blocks) == 2 * self.prefix_one_stream_hbar:
                    self.calibration_blocks = self.calibration_blocks[
                        self.prefix_one_stream_hbar:
                    ]
                    self.calibration_block_costs = self.calibration_block_costs[
                        self.prefix_one_stream_hbar:
                    ]
                    self.calibration_W_state = self._rebuild_state_from_ids(
                        self._calibration_W_order()
                    )

                self.calibration_blocks.append([])
                self.calibration_block_costs.append(0)

        # Algorithm 4 line 11: update best feasible singleton.
        if (
            singleton_state is not None
            and singleton_value > self.best_singleton_value
        ):
            self.best_singleton_state = self._snapshot_state(singleton_state)
            self.best_singleton_value = float(singleton_value)

        # Algorithm 4 lines 12-13: T=TailB(W), C=max{T,best singleton}.
        tail_order = self._tail_b_order(self._calibration_W_order())
        tail_state = self._snapshot_for_ids(tail_order)
        tail_value = self._state_value(tail_state)
        self.calibration_certificate_oracle_calls += 1
        self.oracle_calls_approx += 1

        if self.best_singleton_value > tail_value:
            C_state = self.best_singleton_state
            C_value = self.best_singleton_value
            C_source = "prefix_onestream_singleton"
        else:
            C_state = tail_state
            C_value = tail_value
            C_source = "prefix_onestream_tail"

        if C_value > self.H:
            self.H = float(C_value)
            self.best_calibration_state = self._snapshot_state(C_state)

        # Algorithm 5 lines 8-9.
        if self.H > self.best_value:
            self._set_best_state(
                self.best_calibration_state,
                C_source,
            )

    # ==================================================================
    # Algorithm 5: calibrated geometric guessing
    # ==================================================================

    def _geometric_grid(self, lower, upper):
        lower = float(lower)
        upper = float(upper)
        if lower <= 0.0 or upper < lower:
            return []
        base = 1.0 + self.xi
        log_base = math.log(base)
        j_min = int(math.ceil(
            math.log(lower) / log_base - self._GRID_TOL
        ))
        j_max = int(math.floor(
            math.log(upper) / log_base + self._GRID_TOL
        ))
        if j_max < j_min:
            return []
        return [base ** j for j in range(j_min, j_max + 1)]

    def _prune_below(self, lower, *, after_processing):
        removed = 0
        for guess in list(self.candidates):
            if guess < lower:
                del self.candidates[guess]
                removed += 1
        self.thresholds_removed_total += removed
        if after_processing:
            self.thresholds_pruned_after_processing += removed
        else:
            self.thresholds_pruned_before_processing += removed
        return removed

    def _activate_missing(self, lower, upper):
        born = set()
        for guess in self._geometric_grid(lower, upper):
            if guess not in self.candidates:
                self.candidates[guess] = self._empty_candidate_state
                self.thresholds_created += 1
                born.add(guess)
        self.thresholds_peak = max(
            self.thresholds_peak, len(self.candidates)
        )
        return born

    def process_chunk(self, chunk_record):
        emb = self.compressor._encode_texts(
            [chunk_record["embed_text"]],
            prefix=self.compressor.unit_embedding_prefix,
        )[0].astype(np.float32)
        return self.process_chunk_with_embedding(chunk_record, emb)

    def process_chunk_with_embedding(self, chunk_record, new_emb):
        if self.compacted_to_best:
            raise RuntimeError("Cannot process after final compaction.")

        self._assert_best_sync()
        self.processed_chunks += 1

        cost = int(chunk_record["stream_token_cost"])
        if cost > self.token_budget:
            self.dropped_oversized_chunks += 1
            return {
                "accepted": False,
                "dropped": True,
                "reason": "chunk_cost_exceeds_budget",
                "cost": cost,
            }

        self.feasible_chunks += 1
        self.total_feasible_seen += 1

        if cost > self.beta * self.token_budget + 1e-9:
            # Diagnostic only: do not abort execution when the empirical
            # chunk granularity exceeds the configured theoretical beta cap.
            self.granularity_cap_violations += 1

        new_emb = self._normalize_embedding(new_emb)
        new_idx = self.next_stream_index
        self.next_stream_index += 1

        # Similarities are computed only to currently retained historical
        # elements; discarded raw elements are never revisited.
        sim_lookup = self._sim_new_to_retained(new_emb)
        coverage_features = self._embedding_to_coverage_features(new_emb)

        # Keep the current element temporarily.  End-of-arrival GC removes it if
        # no Algorithm-4/5 state references it.
        self.embedding_store[new_idx] = new_emb.copy()
        self.record_store[new_idx] = chunk_record

        singleton_state = self._singleton_state_for_new(
            new_idx, coverage_features, cost
        )
        self.singleton_oracle_calls += 1
        self.oracle_calls_approx += 1
        singleton_value = (
            0.0 if singleton_state is None else self._state_value(singleton_state)
        )

        # Algorithm 4 is updated first, exactly as Algorithm 5 line 5 requires.
        self._prefix_onestream_update(
            new_idx,
            cost,
            coverage_features,
            sim_lookup,
            singleton_state,
            singleton_value,
        )

        if self.H <= 0.0:
            self._gc_unreferenced_elements()
            self._update_memory_peaks()
            return {
                "accepted": False,
                "dropped": False,
                "reason": "prefix_onestream_H_is_zero",
                "new_idx": new_idx,
            }

        # Algorithm 5 lines 12-13.
        lower = self.best_value / self.gamma
        upper = self.onestream_C0 * self.H / self.eta
        pruned_before = self._prune_below(
            lower, after_processing=False
        )
        born = self._activate_missing(lower, upper)

        accepted_by_any_guess = False
        accepted_guesses = 0

        # COW optimization: equal lineages have equal marginal gains.
        grouped = {}
        for guess in sorted(self.candidates):
            state = self.candidates[guess]
            signature = self._state_signature(state)
            group = grouped.get(signature)
            if group is None:
                group = {"state": state, "guesses": []}
                grouped[signature] = group
                self.state_signature_cache_misses += 1
            else:
                self.state_signature_cache_hits += 1
                if group["state"] is not state:
                    self.candidates[guess] = group["state"]
            group["guesses"].append(guess)

        self.state_signature_unique_states_total += len(grouped)
        self.cow_parent_groups_processed += len(grouped)

        for _, group in grouped.items():
            parent_state = group["state"]
            guesses = group["guesses"]

            (
                marginal_gain,
                coverage_gain,
                diversity_gain,
                new_coverage_mass,
                dpp_update,
            ) = self._marginal_gain_new_parts(
                parent_state,
                new_idx,
                coverage_features,
                sim_lookup,
            )

            accepting_guesses = []
            for guess in guesses:
                self.marginal_oracle_calls += 1
                self.oracle_calls_approx += 1

                if not np.isfinite(marginal_gain):
                    continue

                # Algorithm 5 line 16:
                # Delta >= (alpha v / B) c(u).
                density_condition = (
                    marginal_gain
                    >= (
                        self.density_alpha
                        * float(guess)
                        / float(self.token_budget)
                    )
                    * float(cost)
                )
                budget_condition = (
                    parent_state["current_cost"] + cost
                    <= self.token_budget
                )

                if density_condition and not budget_condition:
                    self.blocked_high_density_events += 1

                if density_condition and budget_condition:
                    accepting_guesses.append(guess)

            if not accepting_guesses:
                continue

            all_accept = len(accepting_guesses) == len(guesses)
            can_mutate_parent_in_place = (
                all_accept and parent_state is not self._empty_candidate_state
            )

            if can_mutate_parent_in_place:
                child_state = parent_state
                self.cow_inplace_group_updates += 1
            else:
                child_state = self._copy_state(
                    parent_state, include_dpp_buffer=True
                )
                self.cow_child_states_created += 1

            ok = self._append_new_to_state(
                state=child_state,
                new_idx=new_idx,
                coverage_features=coverage_features,
                sim_lookup=sim_lookup,
                cost=cost,
                coverage_gain=coverage_gain,
                diversity_gain=diversity_gain,
                new_coverage_mass=new_coverage_mass,
                dpp_update=dpp_update,
            )
            if not ok:
                continue

            for guess in accepting_guesses:
                self.candidates[guess] = child_state

            accepted_count = len(accepting_guesses)
            accepted_by_any_guess = True
            accepted_guesses += accepted_count
            self.accepted_elements += accepted_count

            if accepted_count > 1:
                self.cow_shared_child_assignments += accepted_count
                self.cow_state_copies_avoided += accepted_count - 1

            if self._state_value(child_state) > self.best_value:
                self._set_best_state(
                    child_state,
                    f"calibrated_granular_stream_guess_{accepting_guesses[0]:g}",
                )

        # Algorithm 5 line 20.
        pruned_after = self._prune_below(
            self.best_value / self.gamma,
            after_processing=True,
        )

        self._gc_unreferenced_elements()
        self._assert_best_sync()
        self._update_memory_peaks()

        return {
            "accepted": accepted_by_any_guess,
            "accepted_guesses": accepted_guesses,
            "dropped": False,
            "reason": "algorithm5_calibrated_granular_stream_processed",
            "new_idx": new_idx,
            "activated_now": len(born),
            "new_guesses_processed_current": True,
            "pruned_before": pruned_before,
            "pruned_after": pruned_after,
            "best_value": self.best_value,
            "H": self.H,
            "LB": self.best_value,
        }

    def run_second_pass_augmentation(self):
        # Algorithm 5 is one-pass.  Kept only for the old caller interface.
        return

    # ==================================================================
    # Final compaction and outputs
    # ==================================================================

    def get_selected_indices(self):
        return sorted(set(self.best_state["selected_order"]))

    def compact_to_best_selection(self):
        if self.compacted_to_best:
            return

        self._assert_best_sync()
        self.active_guesses_before_compaction = len(self.candidates)
        self.active_element_copies_before_compaction = self._active_element_copies()
        self.seen_chunks_before_compaction = self.total_feasible_seen

        before = len(self.embedding_store)

        # Only Q is needed after finalization.
        self.candidates.clear()
        self.calibration_blocks = [[]]
        self.calibration_block_costs = [0]
        self.calibration_W_state = self._new_empty_state()
        self.best_calibration_state = self._new_empty_state()
        self.best_singleton_state = self._new_empty_state()
        self.best_singleton_value = 0.0

        deleted_now = self._gc_unreferenced_elements()
        self.deleted_unretained_chunks = deleted_now
        self.deleted_unretained_embeddings = deleted_now

        # Ensure every selected element still exists after GC.
        for idx in self.best_state["selected_order"]:
            if idx not in self.embedding_store or idx not in self.record_store:
                raise RuntimeError("best-state element lost during final compaction")

        self.compacted_to_best = True
        self._assert_best_sync()
        self._update_memory_peaks()

    def get_selected_records_and_embeddings(self):
        ids = self.get_selected_indices()
        seen = set()
        records = []
        rows = []

        for idx in ids:
            record = self.record_store.get(idx)
            emb = self.embedding_store.get(idx)
            if record is None or emb is None:
                continue
            key = chunk_record_dedupe_key(record)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
            rows.append(emb)

        if rows:
            embeddings = np.stack(rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            embeddings = np.zeros(
                (0, int(self.embedding_dim or 0)), dtype=np.float32
            )
        return records, embeddings

    def get_selected_records(self):
        records, _ = self.get_selected_records_and_embeddings()
        return records

    def get_memory(self):
        return render_chunk_records(self.get_selected_records())

    def get_info(self):
        self._assert_best_sync()
        selected_records = self.get_selected_records()
        memory = render_chunk_records(selected_records)
        selected_cost = sum(
            int(chunk["stream_token_cost"])
            for chunk in selected_records
        )

        theoretical_precondition = (
            self.granularity_cap_violations == 0
            and self.passes == 1
            and self.h == 2
        )

        return {
            "algorithm": "calibrated_granular_stream_algorithm_5_faithful_cow",
            "paper_algorithm": "Algorithm 5 CalibratedGranularStream",
            "paper_calibrator": "Algorithm 4 PrefixOneStream(hbar=2)",
            "objective": (
                "lambda_coverage * signed_embedding_feature_saturation "
                "+ lambda_diversity * logdet(I + K_S)"
            ),
            "fixed_submodular_objective": True,
            "coverage_type": "signed_embedding_feature_saturation",
            "coverage_uses_manual_anchors": False,
            "coverage_uses_future_elements": False,
            "coverage_tau": self.coverage_tau,
            "coverage_embedding_dim": self.embedding_dim,
            "coverage_feature_dim": self.coverage_feature_dim,
            "coverage_feature_mapping": (
                "L1-normalized concat(max(e,0), max(-e,0))"
            ),
            "uses_relevance": False,
            "query_dependent": False,
            "question_used_for_compression": False,

            "selected_indices": self.get_selected_indices(),
            "selected_chunks": len(selected_records),
            "seen_chunks": self.total_feasible_seen,
            "selected_stream_token_cost": selected_cost,
            "memory_tokens": self.token_counter(memory),
            "memory_token_budget": self.token_budget,

            "objective_value": self.best_value,
            "best_value": self.best_value,
            "best_source": self.best_source,
            "best_state_value_synchronized": True,
            "coverage_value": float(
                self.best_state.get("coverage_value", 0.0)
            ),
            "diversity_value": float(
                self.best_state.get("diversity_value", 0.0)
            ),
            "weighted_coverage_value": float(
                self.lambda_coverage
                * self.best_state.get("coverage_value", 0.0)
            ),
            "weighted_diversity_value": float(
                self.lambda_diversity
                * self.best_state.get("diversity_value", 0.0)
            ),

            # PDF Algorithm-5 parameters.
            "streaming_xi": self.xi,
            "streaming_eta": self.eta,
            "streaming_beta_cap": self.beta,
            "q_beta": self.q_beta,
            "density_alpha": self.density_alpha,
            "gamma": self.gamma,
            "onestream_C0": self.onestream_C0,
            "prefix_one_stream_hbar": self.prefix_one_stream_hbar,
            "prefix_one_stream_H": self.H,
            "LB": self.best_value,
            "legacy_smkstream_alpha_ignored": self.legacy_alpha,

            "thresholds_alive": len(self.candidates),
            "candidate_sets_alive": len(self.candidates),
            "thresholds_created": self.thresholds_created,
            "thresholds_removed_total": self.thresholds_removed_total,
            "thresholds_pruned_before_processing": (
                self.thresholds_pruned_before_processing
            ),
            "thresholds_pruned_after_processing": (
                self.thresholds_pruned_after_processing
            ),
            "thresholds_peak": self.thresholds_peak,

            "accepted_elements": self.accepted_elements,
            "blocked_high_density_events": self.blocked_high_density_events,
            "processed_chunks": self.processed_chunks,
            "feasible_chunks": self.feasible_chunks,
            "dropped_oversized_chunks": self.dropped_oversized_chunks,
            "granularity_cap_violations": self.granularity_cap_violations,
            "strict_beta_cap": self.strict_beta_cap,
            "theoretical_precondition_satisfied": theoretical_precondition,

            "oracle_calls_approx": self.oracle_calls_approx,
            "singleton_oracle_calls": self.singleton_oracle_calls,
            "calibration_marginal_oracle_calls": (
                self.calibration_marginal_oracle_calls
            ),
            "calibration_certificate_oracle_calls": (
                self.calibration_certificate_oracle_calls
            ),
            "density_marginal_oracle_calls": self.marginal_oracle_calls,
            "numerical_oracle_failures": self.numerical_oracle_failures,

            "single_pass": True,
            "second_pass_augmentation": False,
            "new_guess_processed_current_element": True,
            "streaming_no_question_leakage": True,

            # Element-retention behavior follows the PDF streaming model:
            # raw elements not referenced by maintained states are discarded.
            "bounded_retained_element_store": True,
            "retained_elements_now": len(self.embedding_store),
            "retained_elements_peak": self.retained_elements_peak,
            "gc_deleted_elements_total": self.gc_deleted_elements_total,

            "micro_batch_streaming": True,
            "micro_batch_chunks": 1,
            "online_sequential_processing": True,
            "reencode_all_bank_each_step": False,
            "rebuild_full_similarity_matrix_each_step": False,
            "rerun_batch_selection_each_step": False,

            "compacted_to_best_selection": self.compacted_to_best,
            "deleted_unretained_chunks": self.deleted_unretained_chunks,
            "deleted_unretained_embeddings": self.deleted_unretained_embeddings,
            "retained_chunk_records_after_compaction": len(self.record_store),
            "retained_embeddings_after_compaction": len(self.embedding_store),
            "active_guesses_before_compaction": self.active_guesses_before_compaction,
            "active_element_copies_before_compaction": (
                self.active_element_copies_before_compaction
            ),
            "active_element_copies_peak": self.active_element_copies_peak,
            "element_copies_peak_including_best": (
                self.element_copies_peak_including_best
            ),

            # Preserved engineering optimizations.
            "optimization_similarity_only_to_retained_history": True,
            "sim_to_retained_batch_calls": self.sim_to_retained_batch_calls,
            "sim_to_seen_full_vector_calls": self.sim_to_seen_full_vector_calls,
            "optimization_equivalent_state_marginal_cache": True,
            "optimization_o1_lineage_signature": True,
            "state_signature_cache_hits": self.state_signature_cache_hits,
            "state_signature_cache_misses": self.state_signature_cache_misses,
            "state_signature_unique_states_total": (
                self.state_signature_unique_states_total
            ),
            "signature_transition_count": len(self._signature_transition_ids),
            "optimization_candidate_copy_on_write": True,
            "cow_parent_groups_processed": self.cow_parent_groups_processed,
            "cow_child_states_created": self.cow_child_states_created,
            "cow_inplace_group_updates": self.cow_inplace_group_updates,
            "cow_shared_child_assignments": self.cow_shared_child_assignments,
            "cow_state_copies_avoided": self.cow_state_copies_avoided,
            "unique_candidate_state_peak": self.unique_candidate_state_peak,
            "optimization_dpp_marginal_append_reuse": True,
            "dpp_solve_calls": self.dpp_solve_calls,
            "dpp_solve_reused_for_append": self.dpp_solve_reused_for_append,
            "optimization_dpp_capacity_buffer": True,
            "dpp_buffer_expansions": self.dpp_buffer_expansions,
            "dpp_buffer_allocated_cells_peak": (
                self.dpp_buffer_allocated_cells_peak
            ),
            "optimization_best_state_without_dpp_copy": True,
            "best_state_dpp_matrix_copies": 0,
            "optimization_per_item_gc_collect_removed": True,

            "theoretical_guarantee": (
                "rho_str(beta)-epsilon for xi=eta=epsilon, under the PDF "
                "normalized-monotone-submodular, beta-cap assumptions"
                if theoretical_precondition else None
            ),
        }

class StreamingImportantFactProtector:
    """
    Query-independent, one-pass, bounded important-fact protection channel.

    The main streaming submodular branch keeps the full Stage-1 budget.  This
    class only maintains an auxiliary bounded buffer of high fact-score chunks
    for possible budget-preserving replacement at finalize time.  The current
    Stage-1 embedding is reused directly, so no extra E5 inference is added.
    """

    _NUMBER_RE = re.compile(
        r"(?<!\w)(?:[$€£¥]\s*)?[-+]?\d[\d,]*(?:\.\d+)?%?(?!\w)"
    )

    _TIME_RE = re.compile(
        r"\b(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"today|tomorrow|yesterday|tonight|morning|afternoon|evening|"
        r"week|month|year|hour|minute|"
        r"before|after|earlier|later|previous|previously|recent|recently|"
        r"first|last|latest|next"
        r")\b|"
        r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm|a\.m\.|p\.m\.))?\b|"
        r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
        r"\b\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?\b|"
        r"(?:今天|明天|昨天|今晚|早上|上午|中午|下午|晚上|"
        r"之前|之后|以前|后来|最近|上次|下次|今年|去年|明年|"
        r"\d{1,2}点(?:\d{1,2}分)?|\d{1,2}月\d{1,2}日)",
        flags=re.IGNORECASE,
    )

    _IDENTIFIER_RE = re.compile(
        r"\b(?=[A-Z0-9_-]{4,}\b)(?=[A-Z0-9_-]*[A-Z])"
        r"(?=[A-Z0-9_-]*\d)[A-Z0-9_-]+\b|"
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|"
        r"\b(?:https?://|www\.)\S+|"
        r"\b(?:confirmation|reservation|booking|order|flight|ticket|"
        r"reference|ref|serial|model|account|case|tracking)\s*"
        r"(?:number|no\.?|#|id|code)?\s*[:#-]?\s*[A-Za-z0-9_-]{3,}\b|"
        r"(?:确认号|预订号|订单号|航班号|票号|编号|序列号|型号|账户号|追踪号)"
        r"\s*[:：#-]?\s*[A-Za-z0-9_-]{3,}",
        flags=re.IGNORECASE,
    )

    _UPDATE_RE = re.compile(
        r"\b(?:actually|instead|changed?|changing|updated?|corrected?|"
        r"correction|now|no longer|moved?|moving|switched?|replaced?|"
        r"cancelled?|canceled?|rescheduled?|postponed?|delayed?|"
        r"became|become|stopped?|started?|formerly|used to|"
        r"not anymore|rather than)\b|"
        r"(?:其实|实际|改成|改为|更改|更新|纠正|修正|现在|不再|"
        r"搬到|搬家|换成|替换|取消|延期|延迟|改期|变成|开始|停止|原来|以前)",
        flags=re.IGNORECASE,
    )

    _FACT_CUE_RE = re.compile(
        r"\b(?:my|his|her|their|our)\s+"
        r"(?:name|birthday|address|phone|email|favorite|favourite|"
        r"appointment|reservation|booking|flight|order|job|office|"
        r"school|university|city|location|dog|cat|pet|car|device)\b|"
        r"\b(?:name is|born on|live[sd]? in|work[sd]? at|stud(?:y|ies|ied) at|"
        r"favorite is|favourite is|appointment is|reservation is|"
        r"meeting is|costs?|paid|bought|ordered|booked|scheduled)\b|"
        r"\b(?:dog|cat|pet)'?s?\s+name\s+is\b|"
        r"\b(?:phone|email|address|birthday|reservation|booking|"
        r"appointment|flight|order)\s*(?:is|was|:)\b|"
        r"(?:名字是|生日|地址|电话|邮箱|最喜欢|预约|预订|航班|订单|"
        r"住在|搬到|工作在|就职于|就读于|花了|支付|购买|安排在)",
        flags=re.IGNORECASE,
    )

    _ENTITY_RE = re.compile(
        r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3}\b"
    )
    _TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

    _ENTITY_STOP = {
        "User", "Assistant", "System", "Session", "Date",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "January", "February", "March", "April",
        "May", "June", "July", "August", "September", "October",
        "November", "December", "Today", "Tomorrow", "Yesterday",
    }

    _RARE_STOP = {
        "the", "and", "that", "this", "with", "from", "have", "has",
        "had", "was", "were", "are", "for", "but", "not", "you", "your",
        "user", "assistant", "system", "session", "date", "they", "their",
        "them", "his", "her", "she", "him", "our", "ours", "about",
        "would", "could", "should", "there", "here", "what", "when",
        "where", "which", "who", "why", "how", "into", "onto", "than",
        "then", "been", "being", "will", "just", "like", "really",
    }

    def __init__(self, token_budget, min_score=0.17, cost_exponent=0.25, reference_chunk_tokens=128):
        self.token_budget = max(int(token_budget), 0)
        self.min_score = float(min_score)
        self.cost_exponent = float(cost_exponent)
        self.reference_chunk_tokens = max(int(reference_chunk_tokens), 1)

        self.entries = {}
        self.current_cost = 0

        # Historical lexical DF is updated only after the current chunk is scored.
        self.document_frequency = {}
        self.documents_seen = 0

        self.processed_chunks = 0
        self.eligible_chunks = 0
        self.ever_admitted = 0
        self.evictions = 0
        self.oversized_skips = 0
        self.low_score_skips = 0
        self.invalid_embedding_skips = 0
        self.duplicate_updates = 0
        self.peak_chunks = 0
        self.peak_cost = 0

    @staticmethod
    def _sat(count, scale=1.0):
        count = max(float(count), 0.0)
        scale = max(float(scale), 1e-9)
        return float(1.0 - math.exp(-count / scale))

    def _rare_component(self, text):
        terms = {
            tok.lower()
            for tok in self._TOKEN_RE.findall(text)
            if len(tok) >= 5 and tok.lower() not in self._RARE_STOP
        }
        if not terms:
            return 0.0, terms

        novelty = [
            1.0 / math.sqrt(1.0 + self.document_frequency.get(term, 0))
            for term in terms
        ]
        component = min(
            1.0,
            (sum(novelty) / max(len(novelty), 1))
            * min(len(terms) / 6.0, 1.0),
        )
        return float(component), terms

    def _score_chunk(self, chunk_record):
        # Do not use embed_text: its injected Session/Date header would make
        # every chunk look temporal.
        text = str(
            chunk_record.get("content_text")
            or chunk_record.get("rendered_text")
            or ""
        )

        number_count = len(self._NUMBER_RE.findall(text))
        identifier_present = bool(self._IDENTIFIER_RE.search(text))
        time_present = bool(self._TIME_RE.search(text))
        update_present = bool(self._UPDATE_RE.search(text))
        fact_cue_present = bool(self._FACT_CUE_RE.search(text))

        entities = [
            entity
            for entity in self._ENTITY_RE.findall(text)
            if entity not in self._ENTITY_STOP
        ]
        rarity, terms = self._rare_component(text)

        components = {
            "identifier": 1.0 if identifier_present else 0.0,
            "number": self._sat(number_count, 1.0),
            "time": 1.0 if time_present else 0.0,
            "update": 1.0 if update_present else 0.0,
            "fact_cue": 1.0 if fact_cue_present else 0.0,
            "entity": self._sat(len(entities), 1.5),
            "rarity": rarity,
        }

        score = (
            0.20 * components["identifier"]
            + 0.14 * components["number"]
            + 0.14 * components["time"]
            + 0.18 * components["update"]
            + 0.20 * components["fact_cue"]
            + 0.09 * components["entity"]
            + 0.05 * components["rarity"]
        )
        return float(np.clip(score, 0.0, 1.0)), components, terms

    def _update_document_frequency(self, terms):
        self.documents_seen += 1
        for term in terms:
            self.document_frequency[term] = (
                self.document_frequency.get(term, 0) + 1
            )

    def _priority(self, score, cost):
        reference = max(float(self.reference_chunk_tokens), 1.0)
        normalized_cost = max(float(cost) / reference, 0.25)
        return float(score / (normalized_cost ** self.cost_exponent))

    def process_chunk_with_embedding(self, chunk_record, embedding):
        self.processed_chunks += 1
        score, components, terms = self._score_chunk(chunk_record)
        self._update_document_frequency(terms)

        # Every Stage-1 chunk carries its query-independent continuous fact
        # score, even when it is not retained in the auxiliary protector buffer.
        # This lets Stage-2 use fact_score * query_relevance for all candidates.
        chunk_record["fact_protector_score"] = score
        chunk_record["fact_protector_components"] = dict(components)
        chunk_record["fact_protector_candidate"] = bool(
            score >= self.min_score
        )

        if self.token_budget <= 0:
            self.low_score_skips += 1
            return {
                "protected": False,
                "score": score,
                "reason": "protector_disabled_or_zero_budget",
            }

        cost = max(int(chunk_record.get("stream_token_cost", 1) or 1), 1)
        if cost > self.token_budget:
            self.oversized_skips += 1
            return {
                "protected": False,
                "score": score,
                "reason": "chunk_cost_exceeds_fact_buffer",
            }
        if score < self.min_score:
            self.low_score_skips += 1
            return {
                "protected": False,
                "score": score,
                "reason": "fact_score_below_threshold",
            }

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0 or not np.all(np.isfinite(emb)):
            self.invalid_embedding_skips += 1
            return {
                "protected": False,
                "score": score,
                "reason": "invalid_embedding",
            }

        self.eligible_chunks += 1
        key = chunk_record_dedupe_key(chunk_record)
        priority = self._priority(score, cost)

        protected_record = dict(chunk_record)
        protected_record.update({
            "fact_protected": True,
            "fact_protector_priority": priority,
        })

        previous = self.entries.get(key)
        if previous is not None:
            self.current_cost -= int(previous["cost"])
            self.duplicate_updates += 1

        chunk_order = int(
            chunk_record.get("chunk_order")
            if chunk_record.get("chunk_order") is not None
            else 10**18
        )
        self.entries[key] = {
            "record": protected_record,
            "embedding": emb.copy(),
            "cost": cost,
            "score": score,
            "priority": priority,
            "chunk_order": chunk_order,
        }
        self.current_cost += cost
        self.ever_admitted += 1

        while self.current_cost > self.token_budget and self.entries:
            worst_key = min(
                self.entries,
                key=lambda k: (
                    self.entries[k]["priority"],
                    self.entries[k]["score"],
                    -self.entries[k]["cost"],
                    self.entries[k]["chunk_order"],
                ),
            )
            removed = self.entries.pop(worst_key)
            self.current_cost -= int(removed["cost"])
            self.evictions += 1

        self.peak_chunks = max(self.peak_chunks, len(self.entries))
        self.peak_cost = max(self.peak_cost, self.current_cost)
        retained_now = key in self.entries
        return {
            "protected": retained_now,
            "score": score,
            "priority": priority,
            "reason": "retained" if retained_now else "evicted_by_fact_buffer",
        }

    def get_records_and_embeddings(self):
        ordered = sorted(
            self.entries.values(),
            key=lambda item: item["chunk_order"],
        )
        records = [item["record"] for item in ordered]
        if ordered:
            embeddings = np.stack(
                [item["embedding"] for item in ordered], axis=0
            ).astype(np.float32, copy=False)
        else:
            embeddings = np.zeros((0, 0), dtype=np.float32)
        return records, embeddings

    def get_info(self):
        scores = [item["score"] for item in self.entries.values()]
        return {
            "fact_protector_enabled": True,
            "fact_protector_query_independent": True,
            "fact_protector_single_pass": True,
            "fact_protector_reuses_stream_embedding": True,
            "fact_protector_splits_main_stream_budget": False,
            "fact_protector_buffer_token_budget": self.token_budget,
            "fact_protector_min_score": self.min_score,
            "fact_protector_cost_exponent": self.cost_exponent,
            "fact_protector_processed_chunks": self.processed_chunks,
            "fact_protector_eligible_chunks": self.eligible_chunks,
            "fact_protector_buffer_chunks": len(self.entries),
            "fact_protector_buffer_token_cost": self.current_cost,
            "fact_protector_ever_admitted": self.ever_admitted,
            "fact_protector_evictions": self.evictions,
            "fact_protector_oversized_skips": self.oversized_skips,
            "fact_protector_low_score_skips": self.low_score_skips,
            "fact_protector_invalid_embedding_skips": self.invalid_embedding_skips,
            "fact_protector_duplicate_updates": self.duplicate_updates,
            "fact_protector_peak_chunks": self.peak_chunks,
            "fact_protector_peak_token_cost": self.peak_cost,
            "fact_protector_documents_seen": self.documents_seen,
            "fact_protector_buffer_score_min": min(scores) if scores else None,
            "fact_protector_buffer_score_mean": (
                sum(scores) / len(scores) if scores else None
            ),
            "fact_protector_buffer_score_max": max(scores) if scores else None,
        }


def _retention_scores(
    records,
    embeddings,
    *,
    coverage_tau: float,
    lambda_coverage: float,
    lambda_diversity: float,
):
    n = len(records)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] != n:
        raise ValueError("records/embeddings mismatch in replacement scoring")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("invalid Stage-1 embeddings in replacement scoring")
    emb = emb / norms

    features = np.asarray(
        embedding_to_feature_saturation_features(emb),
        dtype=np.float64,
    )
    total_mass = features.sum(axis=0)
    full_coverage = feature_saturation_coverage_value(total_mass, coverage_tau)
    without_mass = np.maximum(total_mass[None, :] - features, 0.0)
    without_values = (
        np.log1p(float(coverage_tau) * without_mass).sum(axis=1)
        / float(coverage_tau)
    )
    coverage_losses = np.maximum(full_coverage - without_values, 0.0)

    kernel = emb @ emb.T
    kernel = 0.5 * (kernel + kernel.T)
    np.fill_diagonal(kernel, 1.0)
    A = np.eye(n, dtype=np.float64) + kernel
    try:
        L = np.linalg.cholesky(A)
        Linv = solve_triangular(
            L, np.eye(n, dtype=np.float64), lower=True, check_finite=False
        )
        inv_diag = np.sum(Linv * Linv, axis=0)
        inv_diag = np.clip(inv_diag, 1e-300, 1.0)
        diversity_losses = np.maximum(-np.log(inv_diag), 0.0)
    except Exception:
        diversity_losses = np.full(n, np.log(2.0), dtype=np.float64)

    losses = (
        float(lambda_coverage) * coverage_losses
        + float(lambda_diversity) * diversity_losses
    )
    costs = np.asarray(
        [max(int(r.get("stream_token_cost", 1)), 1) for r in records],
        dtype=np.float64,
    )
    return losses, losses / costs


def _merge_fact_protector(
    main_records,
    main_embeddings,
    protected_records,
    protected_embeddings,
    *,
    cfg: MemoryConfig,
):
    main_records = [dict(r) for r in main_records]
    main_embeddings = np.asarray(main_embeddings, dtype=np.float32)
    protected_records = [dict(r) for r in protected_records]
    protected_embeddings = np.asarray(protected_embeddings, dtype=np.float32)

    if main_embeddings.ndim != 2 or main_embeddings.shape[0] != len(main_records):
        raise ValueError("main record/embedding mismatch during fact merge")
    if protected_records and (
        protected_embeddings.ndim != 2
        or protected_embeddings.shape[0] != len(protected_records)
    ):
        raise ValueError("protected record/embedding mismatch during fact merge")

    main_by_key = {
        chunk_record_dedupe_key(record): idx
        for idx, record in enumerate(main_records)
    }
    protected_keys = set()
    overlap = 0
    unique_candidates = []

    for record, emb in zip(protected_records, protected_embeddings):
        key = chunk_record_dedupe_key(record)
        protected_keys.add(key)
        idx = main_by_key.get(key)
        if idx is not None:
            merged = dict(main_records[idx])
            for field in (
                "fact_protected",
                "fact_protector_score",
                "fact_protector_priority",
                "fact_protector_components",
                "fact_protector_candidate",
            ):
                if field in record:
                    merged[field] = record[field]
            merged["fact_protector_overlap_with_submodular"] = True
            main_records[idx] = merged
            overlap += 1
        else:
            injected = dict(record)
            injected["fact_protector_unique_injected"] = True
            unique_candidates.append({
                "record": injected,
                "embedding": np.asarray(emb, dtype=np.float32).copy(),
                "cost": max(int(record.get("stream_token_cost", 1)), 1),
                "priority": float(record.get("fact_protector_priority", 0.0)),
                "score": float(record.get("fact_protector_score", 0.0)),
                "chunk_order": int(record.get("chunk_order", 10**18)),
            })

    unique_candidates.sort(
        key=lambda x: (-x["priority"], -x["score"], x["chunk_order"])
    )
    kept_unique, unique_cost = [], 0
    for item in unique_candidates:
        if unique_cost + item["cost"] <= int(cfg.fact_replacement_budget):
            kept_unique.append(item)
            unique_cost += item["cost"]

    main_cost = sum(
        max(int(r.get("stream_token_cost", 1)), 1) for r in main_records
    )
    overflow = max(main_cost + unique_cost - int(cfg.stream_budget), 0)

    losses, densities = _retention_scores(
        main_records,
        main_embeddings,
        coverage_tau=cfg.feature_coverage_tau,
        lambda_coverage=cfg.stream_lambda_coverage,
        lambda_diversity=cfg.stream_lambda_diversity,
    )
    removable = []
    for idx, record in enumerate(main_records):
        if chunk_record_dedupe_key(record) in protected_keys:
            continue
        removable.append({
            "idx": idx,
            "cost": max(int(record.get("stream_token_cost", 1)), 1),
            "loss": float(losses[idx]),
            "density": float(densities[idx]),
            "chunk_order": int(record.get("chunk_order", 10**18)),
        })
    removable.sort(
        key=lambda x: (x["density"], x["loss"], -x["cost"], x["chunk_order"])
    )

    removed_indices, freed, removed_loss = set(), 0, 0.0
    for item in removable:
        if freed >= overflow:
            break
        removed_indices.add(item["idx"])
        freed += item["cost"]
        removed_loss += item["loss"]

    trimmed = []
    if freed < overflow:
        deficit = overflow - freed
        kept_unique.sort(
            key=lambda x: (x["priority"], x["score"], -x["cost"], -x["chunk_order"])
        )
        recovered = 0
        while kept_unique and recovered < deficit:
            item = kept_unique.pop(0)
            trimmed.append(item)
            recovered += item["cost"]

    final_items = []
    for idx, (record, emb) in enumerate(zip(main_records, main_embeddings)):
        if idx not in removed_indices:
            final_items.append((record, np.asarray(emb, dtype=np.float32).copy()))
    for item in kept_unique:
        final_items.append((item["record"], item["embedding"]))
    final_items.sort(key=lambda x: int(x[0].get("chunk_order", 10**18)))

    final_records = [r for r, _ in final_items]
    if final_items:
        final_embeddings = np.stack([e for _, e in final_items], axis=0).astype(
            np.float32, copy=False
        )
    else:
        dim = main_embeddings.shape[1] if main_embeddings.ndim == 2 else 0
        final_embeddings = np.zeros((0, dim), dtype=np.float32)

    final_cost = sum(
        max(int(r.get("stream_token_cost", 1)), 1) for r in final_records
    )
    if final_cost > int(cfg.stream_budget):
        raise RuntimeError(
            f"Fact-protector merge exceeded stream budget: "
            f"{final_cost} > {cfg.stream_budget}"
        )

    return final_records, final_embeddings, {
        "fact_protector_overlap_with_submodular_chunks": overlap,
        "fact_protector_unique_candidates_before_cap": len(unique_candidates),
        "fact_protector_unique_chunks_injected": len(kept_unique),
        "fact_protector_unique_chunks_trimmed_for_budget": len(trimmed),
        "fact_protector_replaced_submodular_chunks": len(removed_indices),
        "fact_protector_replaced_submodular_token_cost": freed,
        "fact_protector_replaced_stage1_objective_loss_estimate": removed_loss,
        "stage1_candidate_token_cost_after_fact_replacement": final_cost,
    }


@dataclass(slots=True)
class StreamingResult:
    memory: str
    records: list[dict[str, Any]]
    embeddings: np.ndarray
    facility_sim: np.ndarray | None
    dpp_kernel: np.ndarray | None
    info: dict[str, Any]


class StreamingMemoryBank:
    """Dataset-agnostic Stage-1 memory bank.

    Embeddings may be computed in small batches for throughput, but selection is
    applied to chunks strictly in stream order and never uses the question.
    """

    def __init__(self, compressor, token_counter, cfg: MemoryConfig):
        self.compressor = compressor
        self.token_counter = token_counter
        self.cfg = cfg
        self.state = StreamingCovDivState(
            compressor=compressor,
            token_counter=token_counter,
            token_budget=cfg.stream_budget,
            lambda_coverage=cfg.stream_lambda_coverage,
            lambda_diversity=cfg.stream_lambda_diversity,
            coverage_tau=cfg.feature_coverage_tau,
            epsilon=cfg.stream_epsilon,
            beta=cfg.stream_beta,
        )
        self.protector = (
            StreamingImportantFactProtector(
                token_budget=cfg.fact_buffer_budget,
                min_score=cfg.fact_min_score,
                cost_exponent=cfg.fact_cost_exponent,
                reference_chunk_tokens=cfg.chunk_tokens,
            )
            if cfg.fact_protector else None
        )

    def build(self, events: list[SessionEvent]) -> StreamingResult:
        chunks = build_chunks(
            events,
            token_counter=self.token_counter,
            chunk_tokens=self.cfg.chunk_tokens,
            safety_margin=self.cfg.token_safety_margin_per_chunk,
            raise_on_oversized_turn=self.cfg.raise_on_oversized_turn,
        )

        batch_size = max(int(self.cfg.encode_batch_size), 1)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c["embed_text"] for c in batch]
            embeddings = self.compressor._encode_texts(
                texts,
                prefix=self.compressor.unit_embedding_prefix,
            )
            for record, emb in zip(batch, embeddings):
                if self.protector is not None:
                    self.protector.process_chunk_with_embedding(record, emb)
                self.state.process_chunk_with_embedding(
                    chunk_record=record,
                    new_emb=emb,
                )

        self.state.run_second_pass_augmentation()
        self.state.compact_to_best_selection()
        main_records, main_embeddings = (
            self.state.get_selected_records_and_embeddings()
        )
        main_embeddings = np.asarray(main_embeddings, dtype=np.float32)

        if self.protector is not None:
            protected_records, protected_embeddings = (
                self.protector.get_records_and_embeddings()
            )
            records, embeddings, merge_info = _merge_fact_protector(
                main_records,
                main_embeddings,
                protected_records,
                protected_embeddings,
                cfg=self.cfg,
            )
        else:
            records = list(main_records)
            embeddings = main_embeddings
            merge_info = {"fact_protector_enabled": False}

        token_cost = sum(
            max(int(r.get("stream_token_cost", 1)), 1) for r in records
        )
        if token_cost > int(self.cfg.stream_budget):
            raise RuntimeError("Stage-1 candidate pool exceeded stream budget.")

        facility_sim = dpp_kernel = None
        if (
            len(records) > 0
            and embeddings.ndim == 2
            and embeddings.shape[0] == len(records)
            and token_cost > int(self.cfg.query_budget)
        ):
            emb64 = embeddings.astype(np.float64, copy=False)
            dpp_kernel = emb64 @ emb64.T
            dpp_kernel = 0.5 * (dpp_kernel + dpp_kernel.T)
            np.fill_diagonal(dpp_kernel, 1.0)
            facility_sim = np.maximum(dpp_kernel, 0.0).astype(
                np.float32, copy=False
            )
            np.fill_diagonal(facility_sim, 1.0)

        info = dict(self.state.get_info())
        if self.protector is not None:
            info.update(self.protector.get_info())
        info.update(merge_info)
        info.update({
            "stage1_candidate_chunks": len(records),
            "stage1_candidate_token_cost": token_cost,
            "stage1_stream_budget": int(self.cfg.stream_budget),
            "question_used_for_streaming_compression": False,
        })

        return StreamingResult(
            memory=render_chunk_records(records),
            records=records,
            embeddings=embeddings,
            facility_sim=facility_sim,
            dpp_kernel=dpp_kernel,
            info=info,
        )
