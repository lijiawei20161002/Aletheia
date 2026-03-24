"""
Semantic technique verifier — replaces regex pattern matching in
PropagandaAuditPipeline with sentence-transformers cosine similarity.

Uses all-MiniLM-L6-v2 (22MB, fast on CPU) to encode both the propaganda text
and the claimed rhetorical technique descriptions, then checks semantic
alignment to verify whether each claimed technique actually appears in the text.

Advantage over regex: handles paraphrase, novel phrasing, and partial matches
that regex misses — and naturally scales to multilingual content with a swap
to a multilingual encoder.
"""

from __future__ import annotations
from typing import Dict, List, Optional

_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.40  # below this → technique not verified

# Seed descriptions per technique category.
# Multiple seeds per category give the verifier more surface area to match against.
_TECHNIQUE_SEEDS: Dict[str, List[str]] = {
    "appeal to fear": [
        "exaggerating threats to provoke fear rather than rational assessment",
        "danger destruction collapse threat annihilate crisis catastrophe doom",
        "terror extinction survival last chance before it is too late",
    ],
    "us vs them": [
        "creating division between in-group and out-group enemies traitors outsiders",
        "real patriots versus enemies of the people tribal framing",
        "them and us division versus against our kind their kind",
    ],
    "false urgency": [
        "manufactured time pressure act now before it is too late",
        "running out of time urgent immediately right now last chance",
        "final hour window is closing now or never emergency action required",
    ],
    "scapegoating": [
        "blaming a specific group for all problems responsible for suffering",
        "they are destroying ruining killing our way of life fault blame",
        "caused by because of them held responsible for our decline",
    ],
    "appeal to authority": [
        "experts agree scientists prove studies show research confirms data proves",
        "authoritative sources confirm official statement credible expert testimony",
    ],
    "dehumanizing language": [
        "comparing people to animals vermin parasites infestation plague rats cockroaches",
        "subhuman inferior degenerate filth stripping humanity from a group",
    ],
    "manufactured consensus": [
        "everyone knows everybody agrees it is obvious no one denies universally acknowledged",
        "undeniable truth clear to all common knowledge beyond dispute",
    ],
    "appeal to emotion": [
        "emotional manipulation targeting feelings instead of reason pathos",
        "playing on fear anger disgust sadness to bypass rational thought",
    ],
    "false dichotomy": [
        "presenting only two options when many alternatives exist either or",
        "you are with us or against us no middle ground binary choice",
    ],
    "ad hominem": [
        "attacking the person rather than their argument personal attack",
        "discrediting the opponent character assassination instead of engaging ideas",
    ],
    "loaded language": [
        "emotionally charged words carrying implicit judgment political bias",
        "rhetorical loaded terms framing language to trigger emotional reaction",
    ],
    "bandwagon": [
        "everyone is doing it join the majority popular appeal social proof",
        "don't be left behind everybody already knows agrees supports",
    ],
    "glittering generalities": [
        "vague virtuous words that sound positive without specific meaning freedom democracy",
        "values buzzwords patriotism justice fairness with no concrete substance",
    ],
}


class SemanticTechniqueVerifier:
    """
    Verify rhetorical technique claims via semantic similarity.

    Lazy-loads the sentence-transformer model on first use so that importing
    this module is cheap even if sentence-transformers is not installed.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._seed_embeddings: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def _embed(self, texts: List[str]):
        self._load_model()
        return self._model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    def _get_seed_embeddings(self, category: str):
        if category not in self._seed_embeddings:
            seeds = _TECHNIQUE_SEEDS.get(category, [category])
            self._seed_embeddings[category] = self._embed(seeds)
        return self._seed_embeddings[category]

    def _best_category(self, technique: str) -> Optional[str]:
        """Map a free-form technique name to the closest seed category."""
        tech_lower = technique.lower()
        # Fast path: substring match
        for key in _TECHNIQUE_SEEDS:
            if key in tech_lower or tech_lower in key:
                return key
        # Semantic fallback: find closest seed category by cosine similarity
        try:
            from sentence_transformers import util
            tech_emb = self._embed([tech_lower])
            best_key, best_sim = None, -1.0
            for key in _TECHNIQUE_SEEDS:
                seed_embs = self._get_seed_embeddings(key)
                sim = float(util.cos_sim(tech_emb, seed_embs).max())
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
            return best_key if best_sim > 0.35 else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_techniques(
        self,
        text: str,
        claimed_techniques: List[str],
    ) -> Dict:
        """
        Verify claimed rhetorical techniques against the source text.

        Returns
        -------
        {
            "verified": bool,
            "matched_patterns": list[str],        # "technique: score=0.73"
            "unverified_techniques": list[str],
            "scores": dict[str, float],           # raw similarity per technique
        }
        """
        if not claimed_techniques:
            return {
                "verified": False,
                "matched_patterns": [],
                "unverified_techniques": [],
                "scores": {},
            }

        try:
            from sentence_transformers import util
            self._load_model()

            text_emb = self._embed([text])
            matched, unverified, scores = [], [], {}

            for technique in claimed_techniques:
                category = self._best_category(technique)
                if category is None:
                    unverified.append(technique)
                    scores[technique] = 0.0
                    continue

                seed_embs = self._get_seed_embeddings(category)
                sim = float(util.cos_sim(text_emb, seed_embs).max())
                scores[technique] = round(sim, 3)

                if sim >= self.similarity_threshold:
                    matched.append(f"{technique}: score={sim:.2f}")
                else:
                    unverified.append(technique)

            verified = len(matched) >= max(1, len(claimed_techniques) / 2)
            return {
                "verified": verified,
                "matched_patterns": matched,
                "unverified_techniques": unverified,
                "scores": scores,
            }

        except ImportError:
            return self._regex_fallback(text, claimed_techniques)

    def _regex_fallback(self, text: str, claimed_techniques: List[str]) -> Dict:
        """Regex fallback when sentence-transformers is not installed."""
        import re
        text_lower = text.lower()
        fallback_patterns = {
            "appeal to fear":  [r"\b(threat|danger|destroy|collapse|catastroph)\b"],
            "false urgency":   [r"\b(now or never|last chance|act now|too late)\b"],
            "us vs them":      [r"\b(enemy|enemies|they|them|versus|traitor)\b"],
            "scapegoating":    [r"\b(blame|fault|responsible for|caused by)\b"],
            "loaded language": [r"\b(radical|corrupt|elite|invasion|regime)\b"],
        }
        matched, unverified, scores = [], [], {}
        for technique in claimed_techniques:
            tech_lower = technique.lower()
            patterns = next(
                (p for k, p in fallback_patterns.items() if k in tech_lower), []
            )
            found = any(re.search(p, text_lower) for p in patterns)
            scores[technique] = 0.7 if found else 0.2
            if found:
                matched.append(f"{technique}: regex match")
            else:
                unverified.append(technique)
        return {
            "verified": len(matched) >= max(1, len(claimed_techniques) / 2),
            "matched_patterns": matched,
            "unverified_techniques": unverified,
            "scores": scores,
        }
