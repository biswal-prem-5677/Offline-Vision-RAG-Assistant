import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import tempfile
import math
import json

import streamlit as st
import numpy as np
import torch

from config import (
    INDEX_DIR, MIN_IMAGE_SCORE, MIN_TEXT_SCORE,
    MAX_RESULTS, PAGE_SIZE, JUDGE_TOP_N, JUDGE_THRESHOLD,
    UNCERTAIN_THRESHOLD
)
from retrieval import VisionRetriever
from llm import judge_candidates, generate_answer

torch.manual_seed(42)
np.random.seed(42)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── hide default Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── page background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1e1e3f 100%);
    border-right: 1px solid #2d2d5e;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa !important;
}

/* ── hero header ── */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f64f59 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(102,126,234,0.3);
}
.hero-header h1 {
    color: #ffffff !important;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 0;
}

/* ── stat pills in hero ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-top: 18px;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #fff;
    font-weight: 500;
}

/* ── search bar wrapper ── */
.search-wrapper {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

/* ── search input override ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid rgba(167,139,250,0.4) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(255,255,255,0.35) !important;
}

/* ── mode badge banner ── */
.mode-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: 10px;
    margin: 12px 0;
    font-size: 0.9rem;
    font-weight: 500;
}
.mode-text  { background: rgba(167,139,250,0.15); border: 1px solid rgba(167,139,250,0.35); color: #c4b5fd; }
.mode-image { background: rgba(59,130,246,0.15);  border: 1px solid rgba(59,130,246,0.35);  color: #93c5fd; }
.mode-hybrid{ background: rgba(16,185,129,0.15);  border: 1px solid rgba(16,185,129,0.35);  color: #6ee7b7; }

/* ── result count badge ── */
.result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0 18px 0;
}
.result-count-badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* ── image cards ── */
.img-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    margin-bottom: 16px;
}
.img-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(102,126,234,0.25);
    border-color: rgba(167,139,250,0.4);
}
.card-body {
    padding: 10px 12px 12px 12px;
}
.card-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    text-transform: capitalize;
    margin-bottom: 4px;
}
.card-score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 5px;
    margin: 6px 0 4px 0;
    overflow: hidden;
}
.card-score-bar {
    height: 5px;
    border-radius: 999px;
    transition: width 0.4s;
}
.bar-green  { background: linear-gradient(90deg, #10b981, #34d399); }
.bar-yellow { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.bar-red    { background: linear-gradient(90deg, #ef4444, #f87171); }
.card-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
}
.card-score-text {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
}
.badge-high   { color: #10b981; font-size: 0.72rem; font-weight: 600; }
.badge-mid    { color: #f59e0b; font-size: 0.72rem; font-weight: 600; }
.badge-low    { color: #ef4444; font-size: 0.72rem; font-weight: 600; }
.card-filename {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.25);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
}

/* ── section divider ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.3px;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(167,139,250,0.3), transparent);
}

/* ── pagination ── */
.pag-bar {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    padding: 12px 0;
    margin-bottom: 8px;
}
.pag-info {
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── hint / info boxes ── */
.hint-box {
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    color: #c4b5fd;
    font-size: 0.85rem;
    margin: 8px 0;
}
.warn-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    color: #fbbf24;
    font-size: 0.85rem;
    margin: 8px 0;
}
.err-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    color: #f87171;
    font-size: 0.85rem;
    margin: 8px 0;
}

/* ── LLM answer box ── */
.llm-box {
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px;
    padding: 18px 20px;
    color: #d1fae5;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-top: 12px;
}

/* ── sidebar status dot ── */
.online-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* ── sidebar nav divider ── */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.25), transparent);
    margin: 16px 0;
}

/* ── sliders ── */
.stSlider [data-baseweb="slider"] {
    color: #a78bfa;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
    color: #fff;
    border-radius: 8px;
    font-weight: 500;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(167,139,250,0.35) !important;
    border-radius: 10px !important;
    background: rgba(167,139,250,0.04) !important;
}

/* ── expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
}

/* ── empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: rgba(255,255,255,0.35);
}
.empty-state .big-icon { font-size: 3.5rem; margin-bottom: 12px; }
.empty-state .empty-title { font-size: 1.15rem; font-weight: 600; color: rgba(255,255,255,0.5); }
.empty-state .empty-sub   { font-size: 0.85rem; margin-top: 6px; }

/* ── gallery conf chip ── */
.conf-chip {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 3px;
}

/* ── radio & checkbox ── */
.stRadio label, .stCheckbox label {
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Guard: index must exist ────────────────────────────────────────────────────
INDEX_PATH = os.path.join(INDEX_DIR, "config.json")
if not os.path.exists(INDEX_PATH):
    st.markdown("""
    <div class="err-box" style="font-size:1rem; padding:24px;">
        <b>❌ No index found.</b><br><br>
        Build the index first by running in your terminal:<br>
        <code>python build_index.py --images_dir ./sample_images</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_retriever():
    return VisionRetriever(INDEX_DIR)


retriever = get_retriever()
with open(os.path.join(INDEX_DIR, "metadata.json")) as f:
    metadata = json.load(f)

is_stale = retriever.is_stale()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("query_image_path", None),
    ("results", []),
    ("current_page", 0),
    ("gallery_page", 0),
    ("temp_files", []),
    ("llm_answer", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hero header ───────────────────────────────────────────────────────────────
n_classes = len(set(m["caption"] for m in metadata))
st.markdown(f"""
<div class="hero-header">
    <h1>🔍 Vision RAG Assistant</h1>
    <p>Fully offline multimodal image retrieval powered by a custom CNN + FAISS + TinyLlama</p>
    <div class="stat-row">
        <span class="stat-pill">🖼️ {len(metadata)} images indexed</span>
        <span class="stat-pill">🏷️ {n_classes} categories</span>
        <span class="stat-pill">🧠 256-d CNN embeddings</span>
        <span class="stat-pill">🔒 100% Offline</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 4px 0;">
        <span style="font-size:2rem;">🔍</span>
        <div style="font-size:1.1rem; font-weight:700; color:#a78bfa; margin-top:4px;">Vision RAG</div>
        <div style="font-size:0.75rem; color:#64748b;">Offline AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Status
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#6ee7b7; margin-bottom:4px;">
        <span class="online-dot"></span>All models running locally
    </div>
    <div style="font-size:0.78rem; color:#64748b; margin-bottom:8px;">
        Index: <b style="color:#94a3b8;">{len(metadata)} images</b> &nbsp;|&nbsp;
        Classes: <b style="color:#94a3b8;">{n_classes}</b>
    </div>
    """, unsafe_allow_html=True)

    if is_stale:
        st.markdown("""
        <div class="warn-box">
            ⚠️ Index is stale. Run:<br>
            <code>python build_index.py</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Query image ──
    st.markdown("**📁 Query Image**")
    uploaded = st.file_uploader(
        "Upload an image to search",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
        label_visibility="collapsed",
    )

    if uploaded:
        ext = os.path.splitext(uploaded.name)[1].lower() or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state.query_image_path = tmp.name
        st.session_state.temp_files.append(tmp.name)
        st.session_state.current_page = 0
        st.session_state.llm_answer = ""

    if st.session_state.query_image_path and os.path.exists(st.session_state.query_image_path):
        st.image(st.session_state.query_image_path, caption="Active Query Image", use_container_width=True)
        if st.button("✕ Remove image", key="clear_query", use_container_width=True):
            try:
                os.remove(st.session_state.query_image_path)
            except Exception:
                pass
            st.session_state.query_image_path = None
            st.session_state.results = []
            st.session_state.llm_answer = ""
            st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center; padding:14px; color:rgba(255,255,255,0.25);
                    font-size:0.8rem; border:1.5px dashed rgba(167,139,250,0.2);
                    border-radius:10px; margin-top:4px;">
            No image uploaded
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Search options ──
    st.markdown("**⚙️ Search Options**")

    search_mode = st.radio(
        "Mode",
        ["Top-K", "Show All"],
        horizontal=True,
        key="search_mode",
        label_visibility="collapsed",
    )
    st.caption("Top-K: fixed results  |  Show All: adaptive")

    if search_mode == "Top-K":
        k_val = st.slider("Number of results (K)", 1, 50, 12, key="k_slider")
    else:
        k_val = st.slider("Max results", 10, MAX_RESULTS, 100, key="k_max_slider")

    has_query_image = bool(
        st.session_state.query_image_path and os.path.exists(st.session_state.query_image_path)
    )
    if has_query_image:
        alpha = st.slider(
            "Image ↔ Text weight (α)",
            0.0, 1.0, 0.70, 0.05,
            key="alpha_slider",
            help="1.0 = pure image · 0.0 = pure text · 0.7 = balanced hybrid",
        )
    else:
        alpha = 0.0

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Filters ──
    st.markdown("**🎚️ Filters**")
    enable_threshold = st.checkbox("Filter by minimum score", value=False)
    min_score_val = 0.35
    if enable_threshold:
        min_score_val = st.slider("Min score threshold", 0.0, 1.0, 0.40, 0.05)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── LLM options ──
    st.markdown("**🤖 LLM Options**")
    use_judge = st.checkbox(
        "Enable LLM Judge (TinyLlama)",
        value=False,
        help="Re-ranks candidates with a local LLM. Slow but more accurate."
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ How it works"):
        st.markdown("""
**Image search** → CNN encodes query image (256-d) → FAISS cosine search

**Text search** → MiniLM encodes keyword (384-d) → matches class name captions

**Hybrid** → weighted blend of both scores (α controls image weight)

**LLM judge** → TinyLlama re-ranks top results by relevance

All models run **locally**. No internet required.
        """)

# ── Main: search bar ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔎 Search</div>', unsafe_allow_html=True)

query_text = st.text_input(
    "Search",
    value="",
    key="search_text_box",
    placeholder="Type a class name or keyword: rose, bus, cloud, keyboard, butterfly …",
    label_visibility="collapsed",
)

has_query_text = bool(query_text.strip())
has_any_query = has_query_image or has_query_text

# ── Mode banner ──
if has_query_image and has_query_text:
    st.markdown(
        f'<div class="mode-banner mode-hybrid">🔄 <b>Hybrid Search</b> — image + "<i>{query_text.strip()}</i>" &nbsp;(α = {alpha:.2f})</div>',
        unsafe_allow_html=True
    )
elif has_query_image:
    st.markdown(
        '<div class="mode-banner mode-image">🖼️ <b>Image Search</b> — finding visually similar images</div>',
        unsafe_allow_html=True
    )
elif has_query_text:
    st.markdown(
        f'<div class="mode-banner mode-text">🔎 <b>Text Search</b> — searching for "<i>{query_text.strip()}</i>"</div>',
        unsafe_allow_html=True
    )

# ── Retrieve ──────────────────────────────────────────────────────────────────
if has_any_query:
    results = []
    min_score = min_score_val if enable_threshold else None

    with st.spinner("⚡ Searching …"):
        try:
            if has_query_image and has_query_text:
                if search_mode == "Top-K":
                    results = retriever.query_hybrid(
                        st.session_state.query_image_path,
                        text=query_text.strip(), k=k_val,
                        alpha=alpha, min_score=min_score,
                    )
                else:
                    results = retriever.query_all_related(
                        st.session_state.query_image_path,
                        text=query_text.strip(),
                        threshold=min_score_val, max_results=k_val, alpha=alpha,
                    )
            elif has_query_image:
                if search_mode == "Top-K":
                    results = retriever.query_by_image(
                        st.session_state.query_image_path, k=k_val, min_score=min_score,
                    )
                else:
                    results = retriever.query_all_related(
                        st.session_state.query_image_path,
                        text="", threshold=min_score_val, max_results=k_val, alpha=1.0,
                    )
            elif has_query_text:
                results = retriever.query_by_text(
                    query_text.strip(), k=k_val, min_score=min_score,
                )
        except Exception as e:
            st.markdown(f'<div class="err-box">⚠️ Retrieval error: {e}</div>', unsafe_allow_html=True)
            results = []

    # LLM judge
    if use_judge and results:
        query_ctx = {
            "image_path": st.session_state.query_image_path or "",
            "user_text": query_text.strip(),
            "predicted_label": results[0].get("caption", "unknown") if results else "unknown",
        }
        with st.spinner("🤖 LLM Judge re-ranking …"):
            results = judge_candidates(
                query_ctx, results,
                top_n=min(len(results), JUDGE_TOP_N),
                threshold=JUDGE_THRESHOLD,
            )

    st.session_state.results = results
    total = len(results)

    # ── Results header ──
    st.markdown(
        f'<div class="result-header">'
        f'<span style="font-size:1.05rem; font-weight:600; color:#e2e8f0;">Results</span>'
        f'<span class="result-count-badge">{total} found</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if total == 0:
        if enable_threshold:
            st.markdown(
                f'<div class="warn-box">No results with score ≥ {min_score_val:.2f}. '
                f'Try lowering the threshold in the sidebar.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warn-box">No results found. Try different keywords or upload an image.</div>',
                unsafe_allow_html=True,
            )
    else:
        # Low-confidence notice
        if has_query_text and all(r.get("score", 0.0) < 0.60 for r in results):
            st.markdown(
                f'<div class="hint-box">ℹ️ No exact match for "<b>{query_text.strip()}</b>" in this collection. '
                f'Showing closest semantic candidates (CIFAR-100 class names only).</div>',
                unsafe_allow_html=True,
            )
        elif any(r.get("uncertain", False) for r in results):
            st.markdown(
                '<div class="hint-box">ℹ️ CNN confidence is low — showing closest vector matches.</div>',
                unsafe_allow_html=True,
            )

        # ── Pagination ──
        items_per_page = PAGE_SIZE
        total_pages = max(1, math.ceil(total / items_per_page))
        current = st.session_state.current_page
        if current >= total_pages:
            current = 0
            st.session_state.current_page = 0

        if total_pages > 1:
            pc1, pc2, pc3 = st.columns([1, 4, 1])
            with pc1:
                if st.button("◀ Prev", disabled=(current == 0), key="btn_prev", use_container_width=True):
                    st.session_state.current_page -= 1
                    st.rerun()
            with pc2:
                st.markdown(
                    f'<div class="pag-info" style="text-align:center; padding:8px 0;">'
                    f'Page <b>{current + 1}</b> of <b>{total_pages}</b></div>',
                    unsafe_allow_html=True,
                )
            with pc3:
                if st.button("Next ▶", disabled=(current >= total_pages - 1), key="btn_next", use_container_width=True):
                    st.session_state.current_page += 1
                    st.rerun()

        start = current * items_per_page
        end = min(start + items_per_page, total)
        page_results = results[start:end]

        # ── Grid ──
        cols_per_row = 4
        grid_cols = st.columns(cols_per_row)
        for i, item in enumerate(page_results):
            col = grid_cols[i % cols_per_row]
            with col:
                path     = item.get("path", "")
                caption  = item.get("caption", "unknown").replace("_", " ")
                score    = item.get("score", 0.0)
                filename = item.get("filename", "image")
                conf     = item.get("confidence", 0.0)
                is_direct = item.get("direct_match", False)

                pct = int(score * 100)
                if score >= 0.70:
                    bar_cls   = "bar-green"
                    badge_cls = "badge-high"
                    badge_txt = "✦ High"
                elif score >= 0.45:
                    bar_cls   = "bar-yellow"
                    badge_cls = "badge-mid"
                    badge_txt = "◈ Mid"
                else:
                    bar_cls   = "bar-red"
                    badge_cls = "badge-low"
                    badge_txt = "◇ Low"

                direct_tag = " 🎯" if is_direct else ""

                if path and os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="height:140px; display:flex; align-items:center; '
                        f'justify-content:center; background:rgba(255,255,255,0.04); '
                        f'border-radius:10px; color:rgba(255,255,255,0.3);">📷</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div class="card-body" style="padding:0;">
                        <div class="card-label">{caption}{direct_tag}</div>
                        <div class="card-score-bar-bg">
                            <div class="card-score-bar {bar_cls}" style="width:{pct}%;"></div>
                        </div>
                        <div class="card-meta">
                            <span class="card-score-text">{score:.3f}</span>
                            <span class="{badge_cls}">{badge_txt}</span>
                        </div>
                        <div class="card-filename" title="{filename}">{filename}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("🔍 Similar", key=f"res_sim_{start + i}", use_container_width=True):
                    st.session_state.query_image_path = path
                    st.session_state.current_page = 0
                    st.session_state.llm_answer = ""
                    st.rerun()

    # ── LLM Answer ──
    st.markdown('<div class="section-title">💬 Ask the AI</div>', unsafe_allow_html=True)
    with st.expander("Generate an answer from retrieved images with TinyLlama", expanded=False):
        if st.session_state.results:
            q_default = query_text.strip() or "What do these retrieved images represent?"
            user_question = st.text_input(
                "Your question",
                value=q_default,
                key="llm_user_q",
                placeholder="e.g. What categories of images are most common?",
            )
            if st.button("⚡ Generate Answer", key="btn_gen_ans", use_container_width=True):
                with st.spinner("TinyLlama is thinking …"):
                    ans = generate_answer(user_question, st.session_state.results[:10])
                    st.session_state.llm_answer = ans

            if st.session_state.llm_answer:
                st.markdown(
                    f'<div class="llm-box">💡 {st.session_state.llm_answer}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="hint-box">Run a search first, then ask a question about the results.</div>',
                unsafe_allow_html=True,
            )

# ── Gallery (no query active) ────────────────────────────────────────────────
else:
    st.markdown('<div class="section-title">🖼️ Image Gallery</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#64748b; font-size:0.88rem; margin-bottom:16px;">'
        f'Browsing all <b style="color:#94a3b8;">{len(metadata)}</b> indexed images. '
        f'Upload an image or type a keyword above to search.</div>',
        unsafe_allow_html=True,
    )

    gallery_per_page = 16
    total_gallery  = len(metadata)
    total_gpages   = max(1, math.ceil(total_gallery / gallery_per_page))

    g_page = st.session_state.gallery_page
    if g_page >= total_gpages:
        g_page = 0
        st.session_state.gallery_page = 0

    if total_gpages > 1:
        gc1, gc2, gc3 = st.columns([1, 4, 1])
        with gc1:
            if st.button("◀ Prev", disabled=(g_page == 0), key="gal_prev", use_container_width=True):
                st.session_state.gallery_page -= 1
                st.rerun()
        with gc2:
            st.markdown(
                f'<div class="pag-info" style="text-align:center; padding:8px 0;">'
                f'Gallery page <b>{g_page + 1}</b> of <b>{total_gpages}</b></div>',
                unsafe_allow_html=True,
            )
        with gc3:
            if st.button("Next ▶", disabled=(g_page >= total_gpages - 1), key="gal_next", use_container_width=True):
                st.session_state.gallery_page += 1
                st.rerun()

    g_start = g_page * gallery_per_page
    g_end   = min(g_start + gallery_per_page, total_gallery)
    gallery_items = metadata[g_start:g_end]

    gcols = st.columns(4)
    for i, item in enumerate(gallery_items):
        col = gcols[i % 4]
        with col:
            path       = item.get("path", "")
            caption    = item.get("caption", "unknown").replace("_", " ")
            confidence = item.get("confidence", 0.0)
            filename   = item.get("filename", "")

            conf_pct = int(confidence * 100)
            if confidence >= 0.70:
                bar_cls = "bar-green"
            elif confidence >= 0.40:
                bar_cls = "bar-yellow"
            else:
                bar_cls = "bar-red"

            if path and os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.markdown(
                    '<div style="height:140px; display:flex; align-items:center; '
                    'justify-content:center; background:rgba(255,255,255,0.04); '
                    'border-radius:10px; color:rgba(255,255,255,0.3);">📷</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="padding:0 0 4px 0;">
                    <div class="card-label">{caption}</div>
                    <div class="card-score-bar-bg">
                        <div class="card-score-bar {bar_cls}" style="width:{conf_pct}%;"></div>
                    </div>
                    <div class="card-meta">
                        <span class="card-score-text">conf {confidence:.2f}</span>
                    </div>
                    <div class="card-filename" title="{filename}">{filename}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🔍 Search similar", key=f"gal_search_{g_start + i}", use_container_width=True):
                st.session_state.query_image_path = path
                st.session_state.current_page = 0
                st.session_state.llm_answer = ""
                st.rerun()

# ── Cleanup temp files ────────────────────────────────────────────────────────
for f in list(st.session_state.get("temp_files", [])):
    if f and f != st.session_state.query_image_path and os.path.exists(f):
        try:
            os.remove(f)
        except Exception:
            pass
