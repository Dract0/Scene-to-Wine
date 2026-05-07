"""Scene-Based Wine Recommender — inference pipeline (v2).

Loads the trained price model, wine index, and embeddings, and exposes a
single recommend() function that takes a PIL.Image plus optional filters and
returns the top-K vibe-matched, value-reranked wines with LLM-generated
pairing rationales.

This module is framework-agnostic: it takes a PIL.Image, returns a pandas
DataFrame, and has no Streamlit dependency.

v2 changes from v1:
- Reads `numeric_cols` and `median_vintage` from the model bundle so
  inference matches training exactly.
- Per-column category lookup (`KNOWN_CATS_PER_COL`) — fixes a bug where the
  v1 code unioned categories across all 3 OHE columns.
- NaN-safe region resolution (`safe_region`) — `bool(float('nan'))` is True,
  so the v1 `or`-chain let NaN values fall through.
- Wine-age extraction at inference time for candidates that don't have it.
- Parallel explanation generation via ThreadPoolExecutor (~5× faster wall time).
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SCENE_SYSTEM = (
    "You are a sommelier looking at a photo of someone's setting to recommend "
    "wine. Describe the scene in terms relevant to wine selection. "
    "Output JSON only."
)

SCENE_PROMPT = """Look at this image and describe its scene/vibe in terms useful for picking a wine.

Output a JSON object with EXACTLY these keys:
- "setting": short phrase describing the location/situation (e.g., "outdoor patio at sunset")
- "moods": 3-5 mood adjectives
- "season": one of ["spring", "summer", "fall", "winter", "unclear"]
- "formality": one of ["very casual", "casual", "semi-formal", "formal"]
- "time_of_day": one of ["morning", "afternoon", "evening", "night", "unclear"]
- "atmosphere": 3-5 atmosphere adjectives
- "suggested_wine_traits": 5-10 single adjectives describing wines that would suit this scene

Output ONLY the JSON object."""

EXPLAIN_SYSTEM = (
    "You are a sommelier explaining wine pairings in 2 short sentences."
)


# ---------------------------------------------------------------------------
# Wine type filter keywords
# ---------------------------------------------------------------------------

WINE_TYPE_KEYWORDS = {
    "red": [
        "Red", "Pinot Noir", "Cabernet", "Merlot", "Syrah", "Malbec",
        "Zinfandel", "Tempranillo", "Sangiovese", "Grenache", "Shiraz",
        "Nebbiolo", "Bordeaux", "Rhône", "Rhone",
    ],
    "white": [
        "White", "Chardonnay", "Sauvignon Blanc", "Riesling", "Pinot Gris",
        "Pinot Grigio", "Gewürztraminer", "Viognier", "Albariño", "Chenin",
    ],
    "rose": ["Rosé", "Rose"],
    "sparkling": ["Champagne", "Sparkling", "Prosecco", "Cava"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VINTAGE_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def safe_region(row) -> str:
    """Return the most specific non-NaN region label available."""
    for col in ("region_1", "province", "country"):
        val = row.get(col) if hasattr(row, "get") else row[col]
        if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "unknown":
            return str(val).strip()
    return "—"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _pil_to_b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _value_tag(ratio: float) -> str:
    if ratio >= 1.25:
        return "Excellent value"
    if ratio >= 1.10:
        return "Good value"
    if ratio >= 0.90:
        return "Fair price"
    return "Premium price"


def _extract_wine_age(titles: pd.Series, median_vintage: float) -> pd.Series:
    """Extract vintage year from wine titles, compute age relative to 2017."""
    vintages = titles.str.extract(_VINTAGE_RE.pattern)[0].astype(float)
    vintages = vintages.fillna(median_vintage)
    return (2017 - vintages).clip(lower=0)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class WineRecommender:
    """End-to-end wine recommender. Construct once at app startup."""

    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        from anthropic import Anthropic
        from sentence_transformers import SentenceTransformer

        self.artifacts_dir = Path(artifacts_dir)
        self._client = Anthropic()
        self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Load price model bundle
        with open(self.artifacts_dir / "price_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        self._price_model = bundle["model"]
        self._tfidf = bundle["tfidf"]
        self._ohe = bundle["ohe"]
        self._cat_cols = bundle["cat_cols"]
        # v2: numeric_cols and median_vintage are now bundled
        self._numeric_cols = bundle.get("numeric_cols", ["points"])
        self._median_vintage = bundle.get("median_vintage", 2011.0)
        self._best_iter = bundle["best_iter"]
        self.model_results = bundle["results"]
        self.cv_results = bundle.get("cv_results", {})

        # v2: per-column known-category lookup (was a single union before)
        self._known_cats_per_col: dict[str, set] = {
            col: set(cats.tolist())
            for col, cats in zip(self._cat_cols, self._ohe.categories_)
        }

        # Load wine index
        self.wines = pd.read_parquet(self.artifacts_dir / "wine_indexed.parquet")
        if "vibe_profile" in self.wines.columns:
            self.wines["vibe_profile"] = self.wines["vibe_profile"].apply(
                lambda s: json.loads(s) if isinstance(s, str) else s
            )
        # Make sure wine_age is present even if the parquet didn't ship with it
        if "wine_age" not in self.wines.columns:
            self.wines["wine_age"] = _extract_wine_age(
                self.wines["title"], self._median_vintage
            )

        # Embeddings
        self.embeddings = np.load(self.artifacts_dir / "wine_embeddings.npy")
        if self.embeddings.shape[0] != len(self.wines):
            raise ValueError(
                f"Embedding count ({self.embeddings.shape[0]}) does not match "
                f"wine index ({len(self.wines)})"
            )

    # ------------------------------------------------------------------
    # 1. Vision: image → structured scene description
    # ------------------------------------------------------------------

    def image_to_scene(
        self,
        img: Image.Image,
        user_food: Optional[str] = None,
        user_mood_override: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
    ) -> tuple[dict, str]:
        msg = self._client.messages.create(
            model=model,
            max_tokens=500,
            system=SCENE_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _pil_to_b64_jpeg(img),
                        },
                    },
                    {"type": "text", "text": SCENE_PROMPT},
                ],
            }],
        )
        scene = json.loads(_strip_code_fences(msg.content[0].text))

        flat_parts = [
            f"setting: {scene.get('setting', '')}",
            f"moods: {', '.join(scene.get('moods', []))}",
            f"season: {scene.get('season', '')}",
            f"formality: {scene.get('formality', '')}",
            f"time: {scene.get('time_of_day', '')}",
            f"atmosphere: {', '.join(scene.get('atmosphere', []))}",
            f"wine traits: {', '.join(scene.get('suggested_wine_traits', []))}",
        ]
        if user_food:
            flat_parts.append(f"food being served: {user_food}")
        if user_mood_override:
            flat_parts.append(f"desired mood: {user_mood_override}")
        return scene, " | ".join(flat_parts)

    # ------------------------------------------------------------------
    # 2. Retrieval with hard filters
    # ------------------------------------------------------------------

    def retrieve_candidates(
        self,
        scene_text: str,
        max_price: Optional[float] = None,
        wine_types: Optional[Iterable[str]] = None,
        n_candidates: int = 30,
    ) -> pd.DataFrame:
        mask = pd.Series(True, index=self.wines.index)
        if max_price is not None:
            mask &= self.wines["price"] <= max_price
        if wine_types:
            type_mask = pd.Series(False, index=self.wines.index)
            for t in wine_types:
                for kw in WINE_TYPE_KEYWORDS.get(t.lower(), []):
                    type_mask |= self.wines["variety"].str.contains(
                        kw, case=False, na=False
                    )
            mask &= type_mask

        candidate_ids = self.wines.index[mask].tolist()
        if not candidate_ids:
            return pd.DataFrame()

        candidate_embs = self.embeddings[candidate_ids]
        query_emb = self._encoder.encode([scene_text], normalize_embeddings=True)

        k = min(n_candidates, len(candidate_ids))
        local_knn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(candidate_embs)
        distances, idx = local_knn.kneighbors(query_emb)

        chosen_ids = [candidate_ids[i] for i in idx[0]]
        out = self.wines.loc[chosen_ids].copy()
        out["similarity"] = 1 - distances[0]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. ML re-rank by predicted value
    # ------------------------------------------------------------------

    def _predict_log_price(self, candidates: pd.DataFrame) -> np.ndarray:
        cand = candidates.copy()
        # Per-column "other" lumping (v2 fix)
        for col, top_col in [
            ("variety",  "variety_top"),
            ("country",  "country_top"),
            ("province", "province_top"),
        ]:
            known = self._known_cats_per_col[top_col]
            cand[top_col] = cand[col].fillna("unknown").where(
                cand[col].fillna("unknown").isin(known), other="other"
            )
        # wine_age is required for v2; compute it if missing
        if "wine_age" in self._numeric_cols and "wine_age" not in cand.columns:
            cand["wine_age"] = _extract_wine_age(cand["title"], self._median_vintage)

        X_text = self._tfidf.transform(cand["description"])
        X_cat = self._ohe.transform(cand[self._cat_cols])
        X_num = csr_matrix(cand[self._numeric_cols].values)
        X = hstack([X_text, X_cat, X_num]).tocsr()
        return self._price_model.predict(
            X, num_iteration=self._best_iter,
        )

    def rerank_with_value(
        self,
        candidates: pd.DataFrame,
        top_k: int = 5,
        value_weight: float = 0.5,
    ) -> pd.DataFrame:
        cand = candidates.copy()
        pred_log = self._predict_log_price(cand)
        pred_price = np.expm1(pred_log)
        cand["predicted_price"] = pred_price
        cand["value_ratio"] = pred_price / cand["price"].clip(lower=1)

        sim = cand["similarity"]
        sim_norm = (sim - sim.min()) / max(sim.max() - sim.min(), 1e-9)
        val_norm = cand["value_ratio"].rank(pct=True)
        cand["combined_score"] = (
            (1 - value_weight) * sim_norm + value_weight * val_norm
        )
        cand["value_tag"] = cand["value_ratio"].apply(_value_tag)

        return (
            cand.sort_values("combined_score", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # 4. LLM-generated pairing rationale
    # ------------------------------------------------------------------

    def explain(
        self,
        wine_row: pd.Series,
        scene_dict: dict,
        model: str = "claude-haiku-4-5-20251001",
    ) -> str:
        region = safe_region(wine_row)
        prompt = (
            "The user is in this setting:\n"
            f"- Setting: {scene_dict.get('setting', '')}\n"
            f"- Mood: {', '.join(scene_dict.get('moods', []))}\n"
            f"- Formality: {scene_dict.get('formality', '')}\n"
            f"- Atmosphere: {', '.join(scene_dict.get('atmosphere', []))}\n\n"
            "You are recommending this wine:\n"
            f"- Name: {wine_row['title']}\n"
            f"- Variety: {wine_row['variety']}\n"
            f"- Region: {region}\n"
            f"- Description: {wine_row['description']}\n"
            f"- Vibe: {wine_row['vibe_text']}\n\n"
            "In 2 short sentences (no preamble), explain why this wine fits "
            "the setting. Be specific — mention one concrete trait of the wine "
            "and one concrete aspect of the scene."
        )
        msg = self._client.messages.create(
            model=model,
            max_tokens=200,
            system=EXPLAIN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    def explain_parallel(
        self,
        ranked: pd.DataFrame,
        scene_dict: dict,
        max_workers: int = 5,
    ) -> list[str]:
        """v2: parallel explanation generation. ~5× faster wall time."""
        out: list[str] = [""] * len(ranked)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self.explain, ranked.iloc[i], scene_dict): i
                for i in range(len(ranked))
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    out[i] = fut.result()
                except Exception as e:  # noqa: BLE001
                    out[i] = f"(rationale unavailable: {e})"
        return out

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------

    def recommend(
        self,
        img: Image.Image,
        max_price: Optional[float] = None,
        wine_types: Optional[Iterable[str]] = None,
        user_food: Optional[str] = None,
        user_mood_override: Optional[str] = None,
        top_k: int = 5,
        value_weight: float = 0.5,
        progress_cb=None,
    ) -> tuple[dict, pd.DataFrame]:
        def step(i, label):
            if progress_cb:
                progress_cb(i, label)

        step(1, "Analyzing scene…")
        scene, scene_text = self.image_to_scene(img, user_food, user_mood_override)

        step(2, "Retrieving candidate wines…")
        candidates = self.retrieve_candidates(
            scene_text, max_price=max_price, wine_types=wine_types,
        )
        if candidates.empty:
            return scene, candidates

        step(3, "Re-ranking by predicted value…")
        ranked = self.rerank_with_value(
            candidates, top_k=top_k, value_weight=value_weight,
        )

        step(4, "Generating pairing rationales…")
        ranked["explanation"] = self.explain_parallel(ranked, scene)
        return scene, ranked
