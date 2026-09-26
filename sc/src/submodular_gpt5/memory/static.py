import os
import re
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.linalg import solve_triangular


class StaticSubmodularMemoryCompressor:
    """
    Semantic + DPP submodular memory compressor for long-dialogue memory.

    Objective, raw-cost relevance version:

        F(S) =
            lambda_coverage  * facility_location(S)
          + lambda_diversity * logdet(I + K_S)
          + lambda_relevance * sum_{i in S} cost_i * relevance_i

    Constraint:

        sum_{i in S} cost_i <= budget

    PDF GranularSelect density uses raw token cost:

        density(i | S) = Delta F(i | S) / cost_i

    Algorithm 2 uses the raw-token threshold

        tau = 8 * alpha * Gamma / B

    and accepts u exactly when

        f(u | S) >= tau * c(u).

    Relevance modes:

        dense:
            relevance_i = max(cos(e_i, e_q), 0)

        hybrid:
            relevance_i =
                w_dense   * dense_i
              + w_lexical * bm25_i
              + w_entity  * entity_overlap_i
              + w_time    * time_signal_i

            The final relevance_i is fixed before selection and non-negative,
            so the relevance term remains modular and compatible with
            monotone submodular maximization.
    """

    _DIVERSITY_FAILURE_VALUE = -1e18

    def __init__(
        self,
        tokenizer,
        token_budget: int = 16384,
        lambda_coverage: float = 0.45,
        lambda_diversity: float = 0.45,
        lambda_relevance: float = 0.0,
        lambda_fact_query_bonus: float = 0.0,
        semantic_model_name: str = "intfloat/e5-base-v2",
        semantic_device: Optional[str] = None,
        local_files_only: bool = False,
        encode_batch_size: int = 32,
        unit_embedding_prefix: str = "passage: ",
        query_embedding_prefix: str = "query: ",
        relevance_mode: str = "hybrid",
        hybrid_dense_weight: float = 0.55,
        hybrid_lexical_weight: float = 0.30,
        hybrid_entity_weight: float = 0.10,
        hybrid_time_weight: float = 0.05,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        granular_epsilon: float = 0.1,
        dpp_schur_eps: float = 1e-10,
    ):
        self.tokenizer = tokenizer
        self.token_budget = int(token_budget)
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive.")

        self.lambda_coverage = float(lambda_coverage)
        self.lambda_diversity = float(lambda_diversity)
        self.lambda_relevance = float(lambda_relevance)
        self.lambda_fact_query_bonus = float(lambda_fact_query_bonus)

        if self.lambda_coverage < 0:
            raise ValueError("lambda_coverage must be >= 0.")
        if self.lambda_diversity < 0:
            raise ValueError("lambda_diversity must be >= 0.")
        if self.lambda_relevance < 0:
            raise ValueError("lambda_relevance must be >= 0.")
        if self.lambda_fact_query_bonus < 0:
            raise ValueError("lambda_fact_query_bonus must be >= 0.")
        if self.lambda_fact_query_bonus > 0.0 and self.lambda_relevance <= 0.0:
            raise ValueError(
                "lambda_fact_query_bonus requires lambda_relevance > 0."
            )

        # These used to be public constructor parameters even though every valid
        # experiment forced the same values. Keep them as fixed internal invariants.
        self.lambda_cost = 0.0
        self.diversity_method = "dpp"
        self.min_gain = 0.0

        self.semantic_model_name = semantic_model_name
        self.semantic_device = semantic_device
        self.local_files_only = bool(local_files_only)
        self.encode_batch_size = int(encode_batch_size)
        self.unit_embedding_prefix = unit_embedding_prefix
        self.query_embedding_prefix = query_embedding_prefix

        self.relevance_mode = str(relevance_mode).lower()
        if self.relevance_mode not in {"dense", "hybrid"}:
            raise ValueError("relevance_mode must be either 'dense' or 'hybrid'.")

        self.hybrid_dense_weight = float(hybrid_dense_weight)
        self.hybrid_lexical_weight = float(hybrid_lexical_weight)
        self.hybrid_entity_weight = float(hybrid_entity_weight)
        self.hybrid_time_weight = float(hybrid_time_weight)

        hybrid_weights = [
            self.hybrid_dense_weight,
            self.hybrid_lexical_weight,
            self.hybrid_entity_weight,
            self.hybrid_time_weight,
        ]
        if any(w < 0.0 for w in hybrid_weights):
            raise ValueError("Hybrid relevance weights must be non-negative.")
        if self.relevance_mode == "hybrid" and sum(hybrid_weights) <= 0.0:
            raise ValueError("At least one hybrid relevance weight must be positive.")

        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        if self.bm25_k1 <= 0.0:
            raise ValueError("bm25_k1 must be positive.")
        if not (0.0 <= self.bm25_b <= 1.0):
            raise ValueError("bm25_b must be in [0, 1].")

        self.granular_epsilon = float(granular_epsilon)
        if not (0.0 < self.granular_epsilon < 0.5):
            raise ValueError("granular_epsilon must be in (0, 1/2).")

        self.dpp_schur_eps = float(dpp_schur_eps)
        self.semantic_model = self._init_semantic_model()

    # ============================================================
    # Model / tokenization
    # ============================================================

    def _init_semantic_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if self.semantic_device is not None:
            kwargs["device"] = self.semantic_device
        if self.local_files_only:
            kwargs["local_files_only"] = True

        try:
            return SentenceTransformer(self.semantic_model_name, **kwargs)
        except TypeError:
            if self.local_files_only and not os.path.isdir(self.semantic_model_name):
                raise RuntimeError(
                    "Your sentence-transformers version may not support local_files_only. "
                    "Please pass a local model directory or upgrade sentence-transformers."
                )
            kwargs.pop("local_files_only", None)
            return SentenceTransformer(self.semantic_model_name, **kwargs)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _safe_token_costs(self, units: List[str]) -> List[int]:
        return [max(self.count_tokens(u), 1) for u in units]

    def _validate_costs_and_budget(
        self,
        units: List[str],
        token_costs_override: Optional[List[int]],
        token_budget_override: Optional[int],
    ) -> Tuple[List[int], int, bool]:
        effective_budget = self.token_budget if token_budget_override is None else int(token_budget_override)
        if effective_budget <= 0:
            raise ValueError("effective token budget must be positive.")

        if token_costs_override is None:
            return self._safe_token_costs(units) if units else [], effective_budget, False

        if len(token_costs_override) != len(units):
            raise ValueError(
                "token_costs_override must have the same length as units. "
                f"Got {len(token_costs_override)} costs for {len(units)} units."
            )

        return [max(int(c), 1) for c in token_costs_override], effective_budget, True

    def _encode_texts(self, texts: List[str], prefix: str = "") -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if prefix:
            texts = [prefix + t for t in texts]

        embeddings = self.semantic_model.encode(
            texts,
            batch_size=self.encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings

    # ============================================================
    # Precomputation
    # ============================================================

    def _build_semantic_matrices(self, units: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = self._encode_texts(units, prefix=self.unit_embedding_prefix)
        emb64 = embeddings.astype(np.float64, copy=False)

        dpp_kernel = emb64 @ emb64.T
        dpp_kernel = 0.5 * (dpp_kernel + dpp_kernel.T)
        np.fill_diagonal(dpp_kernel, 1.0)

        # Facility location uses non-negative similarities.
        facility_sim = np.maximum(dpp_kernel, 0.0).astype(np.float32)
        np.fill_diagonal(facility_sim, 1.0)

        return embeddings, facility_sim, dpp_kernel

    def prepare_memory(
        self,
        units: List[str],
        token_costs_override: Optional[List[int]] = None,
        token_budget_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        units = list(units)
        token_costs, effective_budget, costs_overridden = self._validate_costs_and_budget(
            units=units,
            token_costs_override=token_costs_override,
            token_budget_override=token_budget_override,
        )

        origin_tokens = sum(token_costs)
        text_origin_tokens = self.count_tokens("\n".join(units)) if units else 0

        prepared: Dict[str, Any] = {
            "units": units,
            "token_costs": token_costs,
            "origin_tokens": origin_tokens,
            "text_origin_tokens": text_origin_tokens,
            "total_units": len(units),
            "token_budget": effective_budget,
            "default_token_budget": self.token_budget,
            "costs_overridden": costs_overridden,
            "embeddings": None,
            "facility_sim": None,
            "dpp_kernel": None,
            "semantic_model": self.semantic_model_name,
            "diversity_method": self.diversity_method,
            "objective_relevance_mode": "raw_cost_weighted_modular",
            "relevance_mode": self.relevance_mode,
        }

        if not units or origin_tokens <= effective_budget:
            return prepared

        embeddings, facility_sim, dpp_kernel = self._build_semantic_matrices(units)
        prepared["embeddings"] = embeddings
        prepared["facility_sim"] = facility_sim
        prepared["dpp_kernel"] = dpp_kernel
        return prepared

    # ============================================================
    # Hybrid modular relevance
    # ============================================================

    def _relevance_config_info(self) -> Dict[str, Any]:
        return {
            "relevance_mode": self.relevance_mode,
            "hybrid_relevance_weights": {
                "dense": self.hybrid_dense_weight,
                "lexical_bm25": self.hybrid_lexical_weight,
                "entity_overlap": self.hybrid_entity_weight,
                "time_signal": self.hybrid_time_weight,
            },
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
        }

    def _normalize_nonnegative_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.maximum(scores, 0.0)

        max_score = float(scores.max()) if scores.size > 0 else 0.0
        if max_score <= 0.0:
            return np.zeros_like(scores, dtype=np.float32)

        return np.clip(scores / max_score, 0.0, 1.0).astype(np.float32)

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """
        Lightweight tokenizer for BM25.

        English:
            word / number tokens.

        Chinese:
            individual CJK characters.

        For better Chinese BM25, you can replace this with jieba.lcut(text).
        """
        if not text:
            return []

        text = text.lower()
        return re.findall(
            r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]",
            text,
        )

    def _bm25_relevance(self, units: List[str], question: str) -> np.ndarray:
        """
        BM25 lexical relevance, normalized to [0, 1].
        This implementation has no external dependency and does not download models.
        """
        n = len(units)
        if n == 0 or not question:
            return np.zeros(n, dtype=np.float32)

        query_tokens = self._tokenize_for_bm25(question)
        if not query_tokens:
            return np.zeros(n, dtype=np.float32)

        docs = [self._tokenize_for_bm25(u) for u in units]
        doc_lens = np.array([len(d) for d in docs], dtype=np.float64)

        avgdl = float(doc_lens.mean()) if n > 0 else 0.0
        if avgdl <= 0.0:
            return np.zeros(n, dtype=np.float32)

        doc_counters = [Counter(d) for d in docs]

        df = Counter()
        for counter in doc_counters:
            for term in counter.keys():
                df[term] += 1

        query_counter = Counter(query_tokens)
        scores = np.zeros(n, dtype=np.float64)

        for term, qtf in query_counter.items():
            dft = df.get(term, 0)
            if dft <= 0:
                continue

            idf = math.log(1.0 + (n - dft + 0.5) / (dft + 0.5))

            for i, counter in enumerate(doc_counters):
                tf = counter.get(term, 0)
                if tf <= 0:
                    continue

                dl = float(doc_lens[i])
                denom = tf + self.bm25_k1 * (
                    1.0 - self.bm25_b + self.bm25_b * dl / avgdl
                )
                if denom <= 0.0:
                    continue

                scores[i] += float(qtf) * idf * (tf * (self.bm25_k1 + 1.0)) / denom

        return self._normalize_nonnegative_scores(scores)

    def _extract_entity_like_terms(self, text: str) -> Set[str]:
        """
        Lightweight entity/date/number extractor.

        It is not a full NER model. It intentionally focuses on terms dense
        embeddings may underweight in long-memory QA:
            names, acronyms, dates, years, numbers, emails, handles.
        """
        if not text:
            return set()

        terms: Set[str] = set()

        month_names = (
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)"
        )

        def add_matches(pattern: str, flags: int = 0) -> None:
            for match in re.findall(pattern, text, flags=flags):
                item = str(match).strip().lower()
                if item:
                    terms.add(item)

        # Emails / handles.
        add_matches(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)
        add_matches(r"@[A-Za-z0-9_]+")

        # ISO / numeric dates.
        add_matches(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b")
        add_matches(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b")

        # Chinese dates.
        add_matches(r"\d{4}年\d{1,2}月\d{1,2}日?")
        add_matches(r"\d{1,2}月\d{1,2}日")

        # English dates.
        add_matches(
            rf"\b{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
            re.IGNORECASE,
        )
        add_matches(
            rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{month_names}(?:\s+\d{{4}})?\b",
            re.IGNORECASE,
        )

        # Years and numbers.
        add_matches(r"\b\d{4}\b")
        add_matches(r"\b\d+(?:\.\d+)?\b")

        # Acronyms / all-caps terms, e.g. LGBTQ, NLP, CNN.
        add_matches(r"\b[A-Z]{2,}[A-Z0-9&+.-]*\b")

        # Capitalized names / phrases, e.g. Caroline, New York.
        add_matches(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

        stop_entities = {
            "what", "when", "where", "who", "whom", "whose", "why", "how",
            "which", "tell", "give", "list", "did", "does", "do", "is",
            "are", "was", "were", "can", "could", "would", "should",
            "please", "thanks", "thank",
        }

        return {t for t in terms if t not in stop_entities}

    def _entity_overlap_relevance(self, units: List[str], question: str) -> np.ndarray:
        """
        Entity-like overlap relevance.

            score_i = |E(q) intersect E(unit_i)| / |E(q)|

        Returns scores in [0, 1].
        """
        n = len(units)
        if n == 0 or not question:
            return np.zeros(n, dtype=np.float32)

        query_entities = self._extract_entity_like_terms(question)
        if not query_entities:
            return np.zeros(n, dtype=np.float32)

        scores = np.zeros(n, dtype=np.float32)
        denom = max(len(query_entities), 1)

        for i, unit in enumerate(units):
            unit_entities = self._extract_entity_like_terms(unit)
            if not unit_entities:
                continue
            scores[i] = float(len(query_entities.intersection(unit_entities))) / denom

        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def _question_has_time_intent(self, question: str) -> bool:
        if not question:
            return False

        q = question.lower()
        time_intent_pattern = (
            r"\bwhen\b|\bdate\b|\bday\b|\bmonth\b|\byear\b|"
            r"\bbefore\b|\bafter\b|\bduring\b|\brecent\b|\brecently\b|"
            r"\blast\b|\bnext\b|\bearlier\b|\blater\b|"
            r"\bsession\b|\bconversation\b|\bturn\b|\btimeline\b|"
            r"什么时候|何时|哪天|几号|日期|时间|哪年|几月|"
            r"之前|之后|以前|以后|会话|第几|最近"
        )
        return re.search(time_intent_pattern, q, flags=re.IGNORECASE) is not None

    def _unit_has_time_marker(self, text: str) -> bool:
        if not text:
            return False

        month_names = (
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)"
        )

        patterns = [
            r"\bsession\s*\d+\b",
            r"\bconversation\s*\d+\b",
            r"\bturn\s*\d+\b",
            r"\bD\d+[:.-]\d+\b",
            r"\bdate\s*[:：]",
            r"\btime\s*[:：]",
            r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
            r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",
            r"\d{4}年\d{1,2}月\d{1,2}日?",
            r"\d{1,2}月\d{1,2}日",
            rf"\b{month_names}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
            rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{month_names}(?:\s+\d{{4}})?\b",
            r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b",
            r"\b\d{4}\b",
            r"日期|时间|会话|第\d+轮",
        ]

        return any(re.search(p, text, flags=re.IGNORECASE) is not None for p in patterns)

    def _time_relevance(self, units: List[str], question: str) -> np.ndarray:
        n = len(units)
        if n == 0 or not question:
            return np.zeros(n, dtype=np.float32)

        if not self._question_has_time_intent(question):
            return np.zeros(n, dtype=np.float32)

        scores = np.zeros(n, dtype=np.float32)
        for i, unit in enumerate(units):
            scores[i] = 1.0 if self._unit_has_time_marker(unit) else 0.0

        return scores

    def _dense_relevance(self, embeddings: np.ndarray, question: str) -> np.ndarray:
        n = embeddings.shape[0]
        if n == 0 or not question:
            return np.zeros(n, dtype=np.float32)

        q_emb = self._encode_texts([question], prefix=self.query_embedding_prefix)[0].astype(np.float32)
        sim = embeddings @ q_emb
        rel = np.maximum(sim, 0.0)

        return np.clip(rel, 0.0, 1.0).astype(np.float32)

    def _compute_relevance(
        self,
        embeddings: np.ndarray,
        units: List[str],
        question: str,
    ) -> np.ndarray:
        n = embeddings.shape[0]
        if n == 0 or not question or self.lambda_relevance <= 0:
            return np.zeros(n, dtype=np.float32)

        dense_rel = self._dense_relevance(embeddings, question)

        if self.relevance_mode == "dense":
            return dense_rel

        lexical_rel = self._bm25_relevance(units, question)
        entity_rel = self._entity_overlap_relevance(units, question)
        time_rel = self._time_relevance(units, question)

        total_weight = (
            self.hybrid_dense_weight
            + self.hybrid_lexical_weight
            + self.hybrid_entity_weight
            + self.hybrid_time_weight
        )

        relevance = (
            self.hybrid_dense_weight * dense_rel
            + self.hybrid_lexical_weight * lexical_rel
            + self.hybrid_entity_weight * entity_rel
            + self.hybrid_time_weight * time_rel
        ) / max(total_weight, 1e-12)

        relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
        relevance = np.maximum(relevance, 0.0)

        return np.clip(relevance, 0.0, 1.0).astype(np.float32)

    # ============================================================
    # DPP logdet marginal gain
    # ============================================================

    def _dpp_marginal_gain(
        self,
        candidate: int,
        dpp_kernel: np.ndarray,
        selected_order: List[int],
        dpp_L: Optional[np.ndarray],
    ) -> float:
        k_cc = 1.0 + float(dpp_kernel[candidate, candidate])
        if not np.isfinite(k_cc) or k_cc <= self.dpp_schur_eps:
            return self._DIVERSITY_FAILURE_VALUE

        if not selected_order:
            return float(np.log(k_cc))

        if dpp_L is None:
            return self._DIVERSITY_FAILURE_VALUE

        v = dpp_kernel[np.ix_(selected_order, [candidate])]
        if not np.all(np.isfinite(v)):
            return self._DIVERSITY_FAILURE_VALUE

        try:
            z = solve_triangular(dpp_L, v, lower=True, check_finite=False)
        except Exception:
            return self._DIVERSITY_FAILURE_VALUE

        if not np.all(np.isfinite(z)):
            return self._DIVERSITY_FAILURE_VALUE

        schur = k_cc - float((z.T @ z)[0, 0])
        if not np.isfinite(schur) or schur <= self.dpp_schur_eps:
            return self._DIVERSITY_FAILURE_VALUE

        return float(np.log(schur))

    def _append_dpp_cholesky(
        self,
        candidate: int,
        dpp_kernel: np.ndarray,
        selected_order: List[int],
        dpp_L: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        k_cc = 1.0 + float(dpp_kernel[candidate, candidate])
        if not np.isfinite(k_cc) or k_cc <= self.dpp_schur_eps:
            return None

        if not selected_order:
            return np.array([[np.sqrt(k_cc)]], dtype=np.float64)

        if dpp_L is None:
            return None

        v = dpp_kernel[np.ix_(selected_order, [candidate])]

        try:
            z = solve_triangular(dpp_L, v, lower=True, check_finite=False)
        except Exception:
            return None

        if not np.all(np.isfinite(z)):
            return None

        schur = k_cc - float((z.T @ z)[0, 0])
        if not np.isfinite(schur) or schur <= self.dpp_schur_eps:
            return None

        m = len(selected_order)
        new_L = np.zeros((m + 1, m + 1), dtype=np.float64)
        new_L[:m, :m] = dpp_L
        new_L[m, :m] = z.ravel()
        new_L[m, m] = np.sqrt(schur)

        return new_L if np.all(np.isfinite(new_L)) else None

    def _rebuild_dpp_cholesky(
        self,
        selected_order: List[int],
        dpp_kernel: np.ndarray,
    ) -> Optional[np.ndarray]:
        if not selected_order:
            return None

        K_S = dpp_kernel[np.ix_(selected_order, selected_order)]
        IK_S = np.eye(len(selected_order), dtype=np.float64) + K_S

        try:
            return np.linalg.cholesky(IK_S)
        except np.linalg.LinAlgError:
            jitter = self.dpp_schur_eps
            for _ in range(5):
                try:
                    return np.linalg.cholesky(
                        IK_S + jitter * np.eye(len(selected_order), dtype=np.float64)
                    )
                except np.linalg.LinAlgError:
                    jitter *= 10.0

        return None

    # ============================================================
    # State operations
    # ============================================================

    def _new_empty_state(self, n: int) -> Dict[str, Any]:
        return {
            "selected": set(),
            "selected_order": [],
            "current_cost": 0,
            "current_facility_cover": np.zeros(n, dtype=np.float32),
            "dpp_L": None,
        }

    def _copy_state_light(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "selected": set(state["selected"]),
            "selected_order": list(state["selected_order"]),
            "current_cost": int(state["current_cost"]),
            "current_facility_cover": state["current_facility_cover"].copy(),
            "dpp_L": None if state["dpp_L"] is None else state["dpp_L"].copy(),
        }

    def _append_item_to_state(
        self,
        state: Dict[str, Any],
        i: int,
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        token_costs: List[int],
    ) -> bool:
        if i in state["selected"]:
            return False

        old_order = list(state["selected_order"])
        new_L = self._append_dpp_cholesky(i, dpp_kernel, old_order, state["dpp_L"])

        if new_L is None:
            new_L = self._rebuild_dpp_cholesky(old_order + [i], dpp_kernel)

        if new_L is None:
            return False

        state["selected"].add(i)
        state["selected_order"].append(i)
        state["current_cost"] += token_costs[i]
        state["current_facility_cover"] = np.maximum(
            state["current_facility_cover"],
            facility_sim[:, i],
        )
        state["dpp_L"] = new_L

        return True

    def _build_state_from_order(
        self,
        selected_order: List[int],
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        token_costs: List[int],
        n: int,
    ) -> Optional[Dict[str, Any]]:
        state = self._new_empty_state(n)

        for i in selected_order:
            if not self._append_item_to_state(state, i, facility_sim, dpp_kernel, token_costs):
                return None

        return state

    def _state_objective_value(
        self,
        state: Dict[str, Any],
        relevance: np.ndarray,
        token_costs: List[int],
        n: int,
        budget: int,
    ) -> float:
        selected_order = state["selected_order"]
        coverage_value = float(state["current_facility_cover"].sum()) / max(n, 1)

        if state["dpp_L"] is None or not selected_order:
            diversity_value = 0.0
        else:
            diag = np.diag(state["dpp_L"])
            if not np.all(np.isfinite(diag)) or np.any(diag <= 0):
                diversity_value = self._DIVERSITY_FAILURE_VALUE
            else:
                diversity_value = float(2.0 * np.log(diag).sum())

        relevance_value = sum(
            token_costs[j] * float(relevance[j])
            for j in selected_order
        )

        return float(
            self.lambda_coverage * coverage_value
            + self.lambda_diversity * diversity_value
            + self.lambda_relevance * relevance_value
        )

    def _state_marginal_gain(
        self,
        state: Dict[str, Any],
        i: int,
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        relevance: np.ndarray,
        token_costs: List[int],
        n: int,
        budget: int,
    ) -> float:
        if i in state["selected"]:
            return -1e18

        current_cover = state["current_facility_cover"]
        new_cover = np.maximum(current_cover, facility_sim[:, i])
        coverage_gain = float((new_cover - current_cover).sum()) / max(n, 1)

        diversity_gain = self._dpp_marginal_gain(
            i,
            dpp_kernel,
            state["selected_order"],
            state["dpp_L"],
        )
        if diversity_gain <= self._DIVERSITY_FAILURE_VALUE / 2:
            return -1e18

        # Raw-cost-weighted relevance: no division by budget.
        relevance_gain = token_costs[i] * float(relevance[i])

        return float(
            self.lambda_coverage * coverage_gain
            + self.lambda_diversity * diversity_gain
            + self.lambda_relevance * relevance_gain
        )

    def _singleton_state(
        self,
        i: int,
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        token_costs: List[int],
        n: int,
        budget: int,
    ) -> Optional[Dict[str, Any]]:
        if token_costs[i] > budget:
            return None

        state = self._new_empty_state(n)
        return state if self._append_item_to_state(state, i, facility_sim, dpp_kernel, token_costs) else None

    # ============================================================
    # PDF Algorithm 1: Calibrate
    # ============================================================

    def _granular_calibrate(
        self,
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        relevance: np.ndarray,
        token_costs: List[int],
        n: int,
        budget: int,
    ) -> Tuple[float, int]:
        """
        Algorithm 1 Calibrate(N, f, c, B).

        R is not constrained to be feasible. Elements with c(u) > B have
        already been conceptually discarded by Algorithm 3, so they are
        skipped here. The returned calibration is exactly Gamma=f(R)/4.
        """
        state = self._new_empty_state(n)
        oracle_calls = 0

        for i in range(n):
            if token_costs[i] > budget:
                continue

            current_value = self._state_objective_value(
                state=state,
                relevance=relevance,
                token_costs=token_costs,
                n=n,
                budget=budget,
            )
            gain = self._state_marginal_gain(
                state=state,
                i=i,
                facility_sim=facility_sim,
                dpp_kernel=dpp_kernel,
                relevance=relevance,
                token_costs=token_costs,
                n=n,
                budget=budget,
            )
            oracle_calls += 1

            if not np.isfinite(gain):
                continue

            # PDF Algorithm 1, line 3:
            # f(u | R) >= (c(u)/B) * f(R)
            rhs = (float(token_costs[i]) / float(budget)) * current_value
            if gain >= rhs:
                self._append_item_to_state(
                    state,
                    i,
                    facility_sim,
                    dpp_kernel,
                    token_costs,
                )

        f_r = self._state_objective_value(
            state=state,
            relevance=relevance,
            token_costs=token_costs,
            n=n,
            budget=budget,
        )
        gamma = max(float(f_r) / 4.0, 0.0)
        return gamma, oracle_calls

    # ============================================================
    # PDF Algorithm 2: DensitySweep
    # ============================================================

    def _granular_density_sweep(
        self,
        gamma: float,
        alpha: float,
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        relevance: np.ndarray,
        token_costs: List[int],
        n: int,
        budget: int,
    ) -> Tuple[Dict[str, Any], int, int]:
        """
        Algorithm 2 DensitySweep(N, f, c, B, Gamma, epsilon, alpha).

        Only the ordered acceptance list A is retained in
        state["selected_order"]. No chain of prefix solution sets is stored.
        """
        state = self._new_empty_state(n)
        oracle_calls = 0
        threshold_rounds = 0

        gamma = float(gamma)
        if gamma <= 0.0 or not np.isfinite(gamma):
            return state, oracle_calls, threshold_rounds

        epsilon = self.granular_epsilon
        if not (0.0 < epsilon < 0.5):
            raise ValueError("PDF GranularSelect requires epsilon in (0, 1/2).")
        if alpha < 1.0:
            raise ValueError("PDF DensitySweep requires alpha >= 1.")

        # PDF Equation (3) / Algorithm 2 line 1.
        tau = 8.0 * float(alpha) * gamma / float(budget)
        terminal_tau = (
            (1.0 - epsilon) * gamma / (math.e * float(budget))
        )

        # PDF Algorithm 2 line 2.
        while tau > terminal_tau:
            threshold_rounds += 1

            # PDF Algorithm 2 line 3: for all u in N \ S.
            for i in range(n):
                if i in state["selected"]:
                    continue
                # Algorithm 3 line 1 conceptually discards these elements.
                if token_costs[i] > budget:
                    continue
                # Algorithm 2 line 4 feasibility test.
                if state["current_cost"] + token_costs[i] > budget:
                    continue

                gain = self._state_marginal_gain(
                    state=state,
                    i=i,
                    facility_sim=facility_sim,
                    dpp_kernel=dpp_kernel,
                    relevance=relevance,
                    token_costs=token_costs,
                    n=n,
                    budget=budget,
                )
                oracle_calls += 1

                if not np.isfinite(gain):
                    continue

                # Algorithm 2 line 4:
                # f(u | S) >= tau * c(u).
                if gain >= tau * float(token_costs[i]):
                    self._append_item_to_state(
                        state,
                        i,
                        facility_sim,
                        dpp_kernel,
                        token_costs,
                    )

            # Algorithm 2 line 6.
            tau *= 1.0 - epsilon

        return state, oracle_calls, threshold_rounds

    # ============================================================
    # PDF Algorithm 3 helpers: Pref(A,b) and one-item rescue
    # ============================================================

    def _granular_prefix_order(
        self,
        acceptance_order: List[int],
        token_costs: List[int],
        budget_level: float,
    ) -> List[int]:
        """
        Pref(A,b): longest prefix of the ordered acceptance list whose
        cumulative token cost is at most b.
        """
        prefix: List[int] = []
        current_cost = 0

        for i in acceptance_order:
            next_cost = current_cost + token_costs[i]
            if next_cost <= budget_level:
                prefix.append(i)
                current_cost = next_cost
            else:
                break

        return prefix

    def _granular_best_one_item_augmentation(
        self,
        prefix_order: List[int],
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        relevance: np.ndarray,
        token_costs: List[int],
        n: int,
        budget: int,
    ) -> Tuple[Dict[str, Any], float, int]:
        """
        Algorithm 3 line 11:
            u* = argmax f(u | P)
        over u not in P that fits in the residual budget.

        If no such element exists, Algorithm 3 adds P itself.
        """
        base_state = self._build_state_from_order(
            selected_order=prefix_order,
            facility_sim=facility_sim,
            dpp_kernel=dpp_kernel,
            token_costs=token_costs,
            n=n,
        )
        if base_state is None:
            raise RuntimeError("Failed to reconstruct a GranularSelect prefix state.")

        remaining_budget = budget - base_state["current_cost"]
        best_i = None
        best_gain = -float("inf")
        oracle_calls = 0

        for i in range(n):
            if i in base_state["selected"]:
                continue
            if token_costs[i] > remaining_budget:
                continue

            gain = self._state_marginal_gain(
                state=base_state,
                i=i,
                facility_sim=facility_sim,
                dpp_kernel=dpp_kernel,
                relevance=relevance,
                token_costs=token_costs,
                n=n,
                budget=budget,
            )
            oracle_calls += 1

            if gain > best_gain:
                best_gain = gain
                best_i = i

        if best_i is None:
            value = self._state_objective_value(
                state=base_state,
                relevance=relevance,
                token_costs=token_costs,
                n=n,
                budget=budget,
            )
            return base_state, value, oracle_calls

        augmented = self._copy_state_light(base_state)
        ok = self._append_item_to_state(
            augmented,
            best_i,
            facility_sim,
            dpp_kernel,
            token_costs,
        )
        if not ok:
            # Numerically defensive fallback: the mathematical kernel should
            # make this unnecessary, but returning P preserves feasibility.
            value = self._state_objective_value(
                state=base_state,
                relevance=relevance,
                token_costs=token_costs,
                n=n,
                budget=budget,
            )
            return base_state, value, oracle_calls

        value = self._state_objective_value(
            state=augmented,
            relevance=relevance,
            token_costs=token_costs,
            n=n,
            budget=budget,
        )
        return augmented, value, oracle_calls

    def _theoretical_monotone_setting(self) -> bool:
        return (
            self.lambda_coverage >= 0.0
            and self.lambda_diversity >= 0.0
            and self.lambda_relevance >= 0.0
            and self.lambda_cost == 0.0
            and self.min_gain == 0.0
            and self.relevance_mode in {"dense", "hybrid"}
            and self.hybrid_dense_weight >= 0.0
            and self.hybrid_lexical_weight >= 0.0
            and self.hybrid_entity_weight >= 0.0
            and self.hybrid_time_weight >= 0.0
            and 0.0 < self.granular_epsilon < 0.5
        )

    # ============================================================
    # PDF Algorithm 3: GranularSelect
    # ============================================================

    def _compress_granular_select(
        self,
        units: List[str],
        token_costs: List[int],
        origin_tokens: int,
        text_origin_tokens: int,
        budget: int,
        costs_overridden: bool,
        facility_sim: np.ndarray,
        dpp_kernel: np.ndarray,
        relevance: np.ndarray,
        question: str,
        return_info: bool,
    ):
        """
        Faithful implementation of PDF Algorithm 3 GranularSelect.

        1. Discard c(u)>B and compute observed beta=max c(u)/B.
        2. Gamma <- Calibrate.
        3. If beta <= beta*_epsilon:
               DensitySweep(alpha=1), return A.
           Else:
               DensitySweep(alpha=1/epsilon);
               evaluate full A plus every checkpoint
               j=-1,0,...,floor(log_{1+epsilon}(1/epsilon)),
               each with the best feasible one-item augmentation.
        """
        n = len(units)
        epsilon = self.granular_epsilon
        if not (0.0 < epsilon < 0.5):
            raise ValueError("PDF GranularSelect requires epsilon in (0, 1/2).")

        # Algorithm 3 line 1: discard c(u)>B, then compute beta.
        feasible_indices = [
            i for i, c in enumerate(token_costs)
            if 0 < c <= budget
        ]
        observed_beta = max(
            (
                float(token_costs[i]) / float(budget)
                for i in feasible_indices
            ),
            default=0.0,
        )

        # PDF Equation (4):
        # beta*_epsilon =
        # 1 - ln(1/(1/2+epsilon)) / (1-epsilon).
        beta_star_epsilon = 1.0 - (
            math.log(1.0 / (0.5 + epsilon))
            / (1.0 - epsilon)
        )

        gamma, calibrate_calls = self._granular_calibrate(
            facility_sim=facility_sim,
            dpp_kernel=dpp_kernel,
            relevance=relevance,
            token_costs=token_costs,
            n=n,
            budget=budget,
        )
        total_oracle_calls = calibrate_calls

        candidates: List[Tuple[float, Dict[str, Any], str]] = []
        checkpoint_count = 0
        rescue_candidate_count = 0

        if observed_beta <= beta_star_epsilon:
            # Algorithm 3 lines 3-5.
            sweep_alpha = 1.0
            sweep_state, sweep_calls, threshold_rounds = (
                self._granular_density_sweep(
                    gamma=gamma,
                    alpha=sweep_alpha,
                    facility_sim=facility_sim,
                    dpp_kernel=dpp_kernel,
                    relevance=relevance,
                    token_costs=token_costs,
                    n=n,
                    budget=budget,
                )
            )
            total_oracle_calls += sweep_calls

            sweep_value = self._state_objective_value(
                state=sweep_state,
                relevance=relevance,
                token_costs=token_costs,
                n=n,
                budget=budget,
            )
            candidates.append(
                (sweep_value, sweep_state, "fine_density_sweep_alpha_1")
            )
            branch = "fine_grained_fast"
        else:
            # Algorithm 3 lines 6-13.
            sweep_alpha = 1.0 / epsilon
            sweep_state, sweep_calls, threshold_rounds = (
                self._granular_density_sweep(
                    gamma=gamma,
                    alpha=sweep_alpha,
                    facility_sim=facility_sim,
                    dpp_kernel=dpp_kernel,
                    relevance=relevance,
                    token_costs=token_costs,
                    n=n,
                    budget=budget,
                )
            )
            total_oracle_calls += sweep_calls

            full_sweep_value = self._state_objective_value(
                state=sweep_state,
                relevance=relevance,
                token_costs=token_costs,
                n=n,
                budget=budget,
            )
            candidates.append(
                (full_sweep_value, sweep_state, "robust_full_sweep")
            )

            acceptance_order = list(sweep_state["selected_order"])

            # Algorithm 3 line 8.
            m = int(
                math.floor(
                    math.log(1.0 / epsilon)
                    / math.log(1.0 + epsilon)
                )
            )

            # Algorithm 3 lines 9-12, including j=-1.
            for j in range(-1, m + 1):
                if j == -1:
                    prefix_order: List[int] = []
                else:
                    budget_level = (
                        epsilon
                        * float(budget)
                        * ((1.0 + epsilon) ** j)
                    )
                    prefix_order = self._granular_prefix_order(
                        acceptance_order=acceptance_order,
                        token_costs=token_costs,
                        budget_level=budget_level,
                    )

                candidate_state, candidate_value, calls = (
                    self._granular_best_one_item_augmentation(
                        prefix_order=prefix_order,
                        facility_sim=facility_sim,
                        dpp_kernel=dpp_kernel,
                        relevance=relevance,
                        token_costs=token_costs,
                        n=n,
                        budget=budget,
                    )
                )
                total_oracle_calls += calls
                checkpoint_count += 1
                rescue_candidate_count += 1
                candidates.append(
                    (
                        candidate_value,
                        candidate_state,
                        f"robust_checkpoint_j_{j}",
                    )
                )

            branch = "robust_checkpoint"

        # Algorithm 3 line 13 (or the single fast-branch candidate).
        best_value, best_state, best_source = max(
            candidates,
            key=lambda x: x[0],
        )

        selected_order = list(best_state["selected_order"])
        # Preserve chronological rendering while keeping the algorithm's
        # acceptance order in selected_order.
        selected_sorted = sorted(selected_order)

        compressed_units = [units[i] for i in selected_sorted]
        compressed_text = "\n".join(compressed_units)
        compressed_tokens = sum(
            token_costs[i] for i in selected_sorted
        )
        compressed_text_tokens = (
            self.count_tokens(compressed_text)
            if compressed_text else 0
        )

        all_costs_equal = (
            len(token_costs) > 0
            and len(set(token_costs)) == 1
        )
        theoretical_monotone_setting = (
            self._theoretical_monotone_setting()
        )
        theoretical_ratio_for_observed_beta = max(
            0.5 - epsilon,
            1.0 - math.exp(
                -(1.0 - epsilon) * (1.0 - observed_beta)
            ),
        )

        info = {
            "selected_indices": selected_sorted,
            "selected_order": selected_order,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "text_origin_tokens": text_origin_tokens,
            "compressed_text_tokens": compressed_text_tokens,
            "selected_units": len(selected_sorted),
            "total_units": len(units),
            "compression_ratio": (
                origin_tokens / compressed_tokens
                if compressed_tokens > 0 else None
            ),
            "compression_rate": (
                compressed_tokens / origin_tokens
                if origin_tokens > 0 else 0.0
            ),
            "token_budget": budget,
            "effective_token_budget": budget,
            "default_token_budget": self.token_budget,
            "costs_overridden": costs_overridden,

            "algorithm": "pdf_algorithm_3_granular_select",
            "paper_algorithm": "Algorithm 3 GranularSelect",
            "paper_calibrate": "Algorithm 1 Calibrate",
            "paper_density_sweep": "Algorithm 2 DensitySweep",

            "semantic_model": self.semantic_model_name,
            "diversity_method": self.diversity_method,
            "query_dependent": (
                bool(question) and self.lambda_relevance > 0
            ),
            "memory_precomputed": True,

            "objective_relevance_mode": "raw_cost_weighted_modular",
            "objective_relevance_formula": (
                "sum_i cost_i * relevance_i; relevance_i is fixed "
                "and non-negative before selection"
            ),
            "density_cost_formula": "DeltaF(i|S) / cost_i",

            "granular_epsilon": epsilon,
            "granularity_beta_observed": observed_beta,
            "beta_star_epsilon": beta_star_epsilon,
            "granular_branch": branch,
            "density_sweep_alpha_used": sweep_alpha,
            "granular_calibrated_gamma": gamma,
            "density_threshold_rounds": threshold_rounds,
            "granular_budget_checkpoints": checkpoint_count,
            "granular_rescue_candidates": rescue_candidate_count,
            "granular_total_candidates": len(candidates),
            "granular_best_candidate_source": best_source,

            "objective_value": best_value,
            "oracle_calls_approx": total_oracle_calls,
            "all_costs_equal": all_costs_equal,
            "theoretical_constraint_type": (
                "cardinality"
                if all_costs_equal else "single_knapsack"
            ),
            "theoretical_monotone_setting": (
                theoretical_monotone_setting
            ),
            "theoretical_setting": theoretical_monotone_setting,
            "theoretical_guarantee": (
                "max{1/2-epsilon, "
                "1-exp(-(1-epsilon)(1-beta))}"
                if theoretical_monotone_setting else None
            ),
            "theoretical_ratio_for_observed_beta": (
                theoretical_ratio_for_observed_beta
                if theoretical_monotone_setting else None
            ),

            # Explicit diagnostics that the original relevance stack was kept.
            "bm25_and_hybrid_relevance_retained": True,
        }

        info.update(self._relevance_config_info())
        return (
            (compressed_text, info)
            if return_info else compressed_text
        )

    # ============================================================
    # Public API
    # ============================================================

    def _base_info_for_trivial_case(
        self,
        units: List[str],
        token_costs: List[int],
        origin_tokens: int,
        text_origin_tokens: int,
        budget: int,
        costs_overridden: bool,
        question: str,
        algorithm: str,
    ) -> Dict[str, Any]:
        all_costs_equal = len(token_costs) > 0 and len(set(token_costs)) == 1
        theoretical_monotone_setting = self._theoretical_monotone_setting()

        info = {
            "selected_indices": list(range(len(units))),
            "selected_order": list(range(len(units))),
            "origin_tokens": origin_tokens,
            "compressed_tokens": origin_tokens,
            "text_origin_tokens": text_origin_tokens,
            "compressed_text_tokens": self.count_tokens("\n".join(units)) if units else 0,
            "selected_units": len(units),
            "total_units": len(units),
            "compression_ratio": 1.0 if units else None,
            "compression_rate": 1.0 if units else 0.0,
            "token_budget": budget,
            "effective_token_budget": budget,
            "default_token_budget": self.token_budget,
            "costs_overridden": costs_overridden,
            "algorithm": algorithm,
            "semantic_model": self.semantic_model_name,
            "diversity_method": self.diversity_method,
            "query_dependent": bool(question) and self.lambda_relevance > 0,
            "memory_precomputed": True,
            "objective_relevance_mode": "raw_cost_weighted_modular",
            "objective_relevance_formula": (
                "sum_i cost_i * relevance_i, where relevance_i is a fixed "
                "non-negative dense/hybrid query score"
            ),
            "density_cost_formula": "DeltaF(i | S) / cost_i",
            "density_threshold_units": "tau is value-per-token and already contains 1/B",
            "density_relevance_effect": (
                "because relevance_gain = cost_i * relevance_i, the relevance "
                "part of density is not divided by token cost"
            ),
            "all_costs_equal": all_costs_equal,
            "theoretical_constraint_type": "cardinality" if all_costs_equal else "single_knapsack",
            "theoretical_monotone_setting": theoretical_monotone_setting,
            "theoretical_setting": theoretical_monotone_setting,
        }

        info.update(self._relevance_config_info())
        return info

    def compress_precomputed(
        self,
        prepared_memory: Dict[str, Any],
        question: str = "",
        return_info: bool = True,
    ):
        units = prepared_memory["units"]
        token_costs = prepared_memory["token_costs"]
        origin_tokens = prepared_memory["origin_tokens"]
        text_origin_tokens = prepared_memory.get("text_origin_tokens", origin_tokens)
        budget = int(prepared_memory.get("token_budget", self.token_budget))
        costs_overridden = bool(prepared_memory.get("costs_overridden", False))

        if budget <= 0:
            raise ValueError("prepared_memory token_budget must be positive.")

        fact_scores = np.asarray(
            prepared_memory.get(
                "fact_scores",
                np.zeros(len(units), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if fact_scores.shape != (len(units),):
            raise ValueError(
                f"fact_scores must have shape ({len(units)},), got {fact_scores.shape}."
            )
        fact_scores = np.clip(
            np.nan_to_num(fact_scores, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        if not units:
            result = (
                "",
                self._base_info_for_trivial_case(
                    [], [], 0, 0, budget, costs_overridden, question, "empty_input"
                ),
            )
            if return_info:
                result[1].update({
                    "lambda_fact_query_bonus": self.lambda_fact_query_bonus,
                    "fact_query_bonus_applied": False,
                })
                return result
            return ""

        if origin_tokens <= budget:
            text = "\n".join(units)
            if return_info:
                info = self._base_info_for_trivial_case(
                    units,
                    token_costs,
                    origin_tokens,
                    text_origin_tokens,
                    budget,
                    costs_overridden,
                    question,
                    "no_compression_needed",
                )
                info.update({
                    "lambda_fact_query_bonus": self.lambda_fact_query_bonus,
                    "fact_query_bonus_applied": False,
                })
                return text, info
            return text

        embeddings = prepared_memory.get("embeddings")
        facility_sim = prepared_memory.get("facility_sim")
        dpp_kernel = prepared_memory.get("dpp_kernel")
        if embeddings is None or facility_sim is None or dpp_kernel is None:
            raise ValueError(
                "prepared_memory does not contain embeddings/facility_sim/dpp_kernel. "
                "Build it with prepare_memory() or provide reused Stage-1 matrices."
            )

        base_relevance = self._compute_relevance(
            embeddings=embeddings,
            units=units,
            question=question,
        )

        fact_interaction = fact_scores * base_relevance
        if self.lambda_fact_query_bonus > 0.0:
            effective_relevance = (
                base_relevance
                + (self.lambda_fact_query_bonus / self.lambda_relevance)
                * fact_interaction
            ).astype(np.float32, copy=False)
        else:
            effective_relevance = base_relevance

        result = self._compress_granular_select(
            units=units,
            token_costs=token_costs,
            origin_tokens=origin_tokens,
            text_origin_tokens=text_origin_tokens,
            budget=budget,
            costs_overridden=costs_overridden,
            facility_sim=facility_sim,
            dpp_kernel=dpp_kernel,
            relevance=effective_relevance,
            question=question,
            return_info=return_info,
        )

        if not return_info:
            return result

        text, info = result
        info = dict(info)
        info.update({
            "lambda_fact_query_bonus": self.lambda_fact_query_bonus,
            "fact_query_bonus_applied": bool(
                self.lambda_fact_query_bonus > 0.0 and bool(question)
            ),
            "fact_score_nonzero_candidates": int(np.count_nonzero(fact_scores > 0.0)),
            "fact_query_interaction_nonzero": int(
                np.count_nonzero(fact_interaction > 0.0)
            ),
        })
        return text, info

    def compress(
        self,
        units: List[str],
        question: str = "",
        return_info: bool = True,
        token_costs_override: Optional[List[int]] = None,
        token_budget_override: Optional[int] = None,
    ):
        prepared = self.prepare_memory(
            units=units,
            token_costs_override=token_costs_override,
            token_budget_override=token_budget_override,
        )

        return self.compress_precomputed(
            prepared_memory=prepared,
            question=question,
            return_info=return_info,
        )

