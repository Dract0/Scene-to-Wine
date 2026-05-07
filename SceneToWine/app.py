"""Scene-Based Wine Recommender — Streamlit frontend.

Run locally:
    export ANTHROPIC_API_KEY=sk-...
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    Push this repo to GitHub, connect at https://share.streamlit.io,
    and add ANTHROPIC_API_KEY in the app's Secrets settings.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ----- Page config -----
st.set_page_config(
    page_title="Scene-Based Wine Recommender",
    page_icon="🍷",  # browser tab only; not shown in the page itself
    layout="wide",
)

# ----- API key resolution -----
# Order of precedence: Streamlit secrets → environment variable → manual input.
# We check whether secrets.toml actually exists before touching st.secrets to
# avoid the "No secrets found" banner Streamlit otherwise renders.
def _resolve_api_key():
    secrets_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    if any(p.exists() for p in secrets_paths):
        try:
            return st.secrets["ANTHROPIC_API_KEY"]
        except Exception:  # noqa: BLE001
            pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    return None


# ----- Cached resource loader -----
@st.cache_resource(show_spinner="Loading recommender (one-time, ~30s)…")
def load_recommender():
    from pipeline import WineRecommender
    return WineRecommender(artifacts_dir="artifacts")


# ============================================================================
# UI
# ============================================================================

st.title("Scene-Based Wine Recommender")
st.markdown(
    "Upload an image of a setting to receive wine recommendations. "
    "The system extracts scene attributes with a vision model, retrieves "
    "candidates from a curated wine corpus using sentence embeddings, "
    "and re-ranks results using a trained price-prediction model to "
    "surface the strongest matches by predicted value."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    api_key = _resolve_api_key()
    if not api_key:
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            help="Required for vision and explanation calls. For deployment, "
                 "set this in Streamlit secrets.",
        )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.subheader("Filters")

    max_price = st.slider("Maximum price (USD)", 10, 200, 60, step=5)
    wine_type_choice = st.multiselect(
        "Wine type",
        options=["red", "white", "rose", "sparkling"],
        default=[],
        help="Leave empty to consider all wine types.",
    )
    user_food = st.text_input(
        "Food pairing (optional)",
        placeholder="e.g., grilled steak, charcuterie board",
    )
    user_mood = st.text_input(
        "Mood preference (optional)",
        placeholder="e.g., celebratory, contemplative",
    )

    st.divider()
    st.subheader("Ranking")

    value_weight = st.slider(
        "Re-ranking weight",
        0.0, 1.0, 0.5, 0.1,
        help="0.0 = rank by scene similarity only · "
             "0.5 = balanced · "
             "1.0 = rank by predicted value only",
    )
    top_k = st.slider("Number of recommendations", 3, 7, 5)

# Block main content until API key is available
if not api_key:
    st.info("Provide an Anthropic API key in the sidebar to continue.")
    st.stop()

# Load recommender (cached after first call)
try:
    rec = load_recommender()
except FileNotFoundError as e:
    st.error(
        "Required artifacts not found. Ensure `artifacts/` contains "
        "`price_model.pkl`, `wine_indexed.parquet`, and `wine_embeddings.npy`.\n\n"
        f"{e}"
    )
    st.stop()

# --- Main content ---
left, right = st.columns([1, 1])

with left:
    st.subheader("Scene image")
    uploaded = st.file_uploader(
        "Upload an image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)

with right:
    st.subheader("Recommendations")

    if not uploaded:
        st.markdown(":grey[Upload an image to generate recommendations.]")
    else:
        if st.button("Generate recommendations", type="primary",
                     use_container_width=True):
            progress = st.progress(0, text="Initializing…")

            def cb(stage_idx, label):
                statuses = {
                    1: "Analyzing scene…",
                    2: "Retrieving candidate wines…",
                    3: "Re-ranking by predicted value…",
                    4: "Generating pairing rationales…",
                }
                progress.progress(stage_idx / 4, text=statuses.get(stage_idx, label))

            try:
                scene, results = rec.recommend(
                    img,
                    max_price=float(max_price),
                    wine_types=wine_type_choice or None,
                    user_food=user_food or None,
                    user_mood_override=user_mood or None,
                    top_k=top_k,
                    value_weight=value_weight,
                    progress_cb=cb,
                )
                progress.empty()
            except Exception as e:
                progress.empty()
                st.error(f"Recommendation failed: {e}")
                st.stop()

            if results.empty:
                st.warning(
                    "No candidates passed the configured filters. "
                    "Consider raising the maximum price or removing the "
                    "wine-type restriction."
                )
                st.stop()

            st.session_state["scene"] = scene
            st.session_state["results"] = results

# --- Results (rendered outside the button block so they persist) ---
if "results" in st.session_state and uploaded:
    scene = st.session_state["scene"]
    results = st.session_state["results"]

    st.divider()

    with st.expander("Scene analysis", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Setting:** {scene.get('setting', '')}")
            st.markdown(f"**Formality:** {scene.get('formality', '')}")
            st.markdown(f"**Season:** {scene.get('season', '')}")
            st.markdown(f"**Time of day:** {scene.get('time_of_day', '')}")
        with col_b:
            st.markdown(f"**Mood:** {', '.join(scene.get('moods', []))}")
            st.markdown(f"**Atmosphere:** {', '.join(scene.get('atmosphere', []))}")
            st.markdown(
                f"**Suggested wine attributes:** "
                f"{', '.join(scene.get('suggested_wine_traits', []))}"
            )

    for _, row in results.iterrows():
        # NaN-safe region lookup
        region = None
        for col in ("region_1", "province", "country"):
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                region = str(v).strip()
                break
        region = region or "—"

        tag = row["value_tag"]
        tag_color = {
            "Excellent value": "#1d9e75",
            "Good value":      "#639922",
            "Fair price":      "#7a7a72",
            "Premium price":   "#993556",
        }.get(tag, "#7a7a72")

        with st.container(border=True):
            header_l, header_r = st.columns([3, 1])
            with header_l:
                st.markdown(f"### {row['title']}")
                st.markdown(f":grey[{row['variety']} · {region}]")
            with header_r:
                st.markdown(
                    f"<div style='text-align:right; padding-top:0.25rem;'>"
                    f"<span style='background-color:{tag_color}; color:white; "
                    f"padding:0.25rem 0.6rem; border-radius:4px; "
                    f"font-size:0.85rem; font-weight:500;'>"
                    f"{tag}</span></div>",
                    unsafe_allow_html=True,
                )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Price (USD)", f"${row['price']:.0f}")
            with c2:
                st.metric("Rating", f"{int(row['points'])} pts")
            with c3:
                st.metric(
                    "Predicted price",
                    f"${row['predicted_price']:.0f}",
                    delta=f"{(row['value_ratio'] - 1) * 100:+.0f}% vs actual",
                )
            with c4:
                st.metric("Match score", f"{row['similarity'] * 100:.0f}%")

            st.markdown(f"**Pairing rationale:** {row['explanation']}")

            with st.expander("Tasting notes"):
                st.markdown(row["description"])

# --- Methodology panel (sidebar, expandable) ---
with st.sidebar:
    st.divider()
    with st.expander("Methodology"):
        st.markdown(
            "**Component A — Price prediction model.** "
            "LightGBM regression predicting wine price from description "
            "(TF-IDF unigrams and bigrams), variety, country, province, "
            "rating, and wine age. Reviewer price-vocabulary is filtered "
            "from the text features. Evaluated on a held-out 15% test set "
            "with 5-fold cross-validation."
        )
        try:
            metrics = rec.model_results
            df = pd.DataFrame({
                "Model": list(metrics["MAE_$"].keys()),
                "MAE (USD)": [round(v, 2) for v in metrics["MAE_$"].values()],
                "RMSE (USD)": [round(v, 2) for v in metrics["RMSE_$"].values()],
                "R² (log)": [round(v, 3) for v in metrics["R2_log"].values()],
            })
            st.dataframe(df, hide_index=True, use_container_width=True)
        except Exception:
            pass

        st.markdown(
            "**Component B — Semantic retrieval.** "
            "A vision model converts the scene image to structured "
            "attributes. Each wine in the index is paired with an "
            "LLM-generated profile describing occasions, moods, food "
            "pairings, and stylistic adjectives. Both sides are encoded "
            "with `sentence-transformers/all-MiniLM-L6-v2`; KNN retrieves "
            "the top 30 nearest candidates."
        )
        st.markdown(
            "**Component C — Decision layer.** "
            "Candidates are re-ranked using a weighted combination of "
            "scene similarity and the price model's predicted-vs-actual "
            "value ratio. The top results are returned with a value "
            "category and a generated pairing rationale."
        )
