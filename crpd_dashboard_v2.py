# crpd_dashboard_v2.py
# Generalized Streamlit dashboard for CRPD text analysis (Article 13 / 9 / 24 / custom)
# - Upload or auto-load CRPD_reports.csv
# - Filter by doc_type and article keywords
# - Global & country word clouds, regional mentions, regional sentiment
# - Clean UI (tabs, metrics, Plotly), CSV exports

import re
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================
# Page Config & Theming
# =========================
st.set_page_config(
    page_title="CRPD Article Explorer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle styling
st.markdown(
    """
    <style>
    .metric-card {background: #f7f9fc; border: 1px solid #e6ecf5; border-radius: 12px; padding: 14px;}
    .section-title {color:#155e75; border-left: 4px solid #14b8a6; padding-left:10px; margin: 8px 0 16px 0;}
    .caption {color:#475569; font-size:0.9rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background: #f1f5f9; border-radius: 8px; padding: 8px 12px;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background: #e0f2fe; border: 1px solid #38bdf8;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Utilities
# =========================

def load_data(uploaded_file: io.BytesIO | None) -> pd.DataFrame | None:
    """Load CSV from uploader or fallback to CRPD_reports.csv if present."""
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        # Fallback to local file if exists
        return pd.read_csv("CRPD_reports.csv")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Minimal region mapping; extend as needed
REGION_MAP = {
    "Kenya": "Africa", "South Africa": "Africa", "Ghana": "Africa", "Nigeria": "Africa",
    "Nepal": "Asia-Pacific", "India": "Asia-Pacific", "Bangladesh": "Asia-Pacific",
    "Brazil": "Latin America", "Chile": "Latin America", "Peru": "Latin America", "Mexico": "Latin America",
    "Germany": "Europe", "France": "Europe", "Spain": "Europe", "Italy": "Europe",
    "Canada": "North America", "United States": "North America"
}
# Manually set dictionary of CRPD articles and their keyword presets
ARTICLE_PRESETS = {
    "Article 13 — Access to Justice": ["justice"],
    "Article 9 — Accessibility": ["accessibility", "accessible"],
    "Article 24 — Education": ["education", "school", "inclusive education"],
    "Custom": [],
}


def ensure_region_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Region" not in df.columns:
        df["Region"] = df.get("country", pd.Series([None]*len(df))).map(REGION_MAP)
    df["Region"] = df["Region"].fillna("Unmapped")
    return df


def count_mentions(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str) or not keywords:
        return 0
    total = 0
    for kw in keywords:
        try:
            total += len(re.findall(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE))
        except re.error:
            # Fallback to literal contains if regex fails
            total += text.lower().count(kw.lower())
    return total


def compute_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(TextBlob(text).sentiment.polarity)


def make_wordcloud(text: str, width: int = 1200, height: int = 600):
    if not isinstance(text, str) or not text.strip():
        return None
    cloud = WordCloud(width=width, height=height, background_color="white", max_words=150).generate(text)
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return fig


# =========================
# Sidebar — Controls
# =========================
st.sidebar.header("Configuration ⚙️")
uploaded = st.sidebar.file_uploader("Upload CRPD CSV (optional)", type=["csv"], help="If omitted, the app will try CRPD_reports.csv in the working directory.")

# Load data ASAP so other widgets can depend on it
_df = load_data(uploaded)
if _df is None or _df.empty:
    st.stop()

# Light schema hint
expected_cols = {"doc_type", "text", "country"}
missing = expected_cols - set(_df.columns)
if missing:
    st.warning(f"Your data is missing expected columns: {sorted(missing)}. The app will run with reduced features.")

# Doc type filter
doc_types = sorted(_df.get("doc_type", pd.Series(dtype=str)).dropna().unique().tolist())
default_doc_types = [dt for dt in doc_types if str(dt).lower() in {"concluding observations", "concluding observation"}] or doc_types
selected_doc_types = st.sidebar.multiselect("Document types", options=doc_types, default=default_doc_types)

# Article preset / keywords
preset = st.sidebar.selectbox("Article focus", list(ARTICLE_PRESETS.keys()), index=0)
keywords = ARTICLE_PRESETS[preset]
custom_kw = st.sidebar.text_input("Keywords (comma-separated)", value=", ".join(keywords) if keywords else "justice")
keywords = [k.strip() for k in custom_kw.split(",") if k.strip()]

min_len = st.sidebar.slider("Minimum text length (chars)", 0, 2000, 0, 50)

# Country filter (optional)
all_countries = sorted(_df.get("country", pd.Series(dtype=str)).dropna().unique().tolist())
selected_countries = st.sidebar.multiselect("Countries (optional)", options=all_countries)

# =========================
# Data Filtering & Feature Columns
# =========================
df = _df.copy()
if selected_doc_types:
    df = df[df.get("doc_type", "").isin(selected_doc_types)]

if min_len:
    df = df[df.get("text", "").astype(str).str.len() >= min_len]

if selected_countries:
    df = df[df.get("country", "").isin(selected_countries)]

# Filter by keywords (match if any keyword appears)
if keywords:
    patt = re.compile(r"|".join([rf"\b{re.escape(k)}\b" for k in keywords]), flags=re.IGNORECASE)
    df = df[df.get("text", "").astype(str).str.contains(patt)]

# Add helper columns
if "text" in df.columns:
    df["Mentions"] = df["text"].astype(str).apply(lambda t: count_mentions(t, keywords))
    df["Sentiment"] = df["text"].astype(str).apply(compute_sentiment)

# Region enrichment
df = ensure_region_column(df)

# Guard empty state after filtering
if df.empty:
    st.info("No rows match your filters. Try broadening the document types, countries, or keywords.")
    st.stop()

# =========================
# Header & KPIs
# =========================
st.title("CRPD Article Explorer")
st.caption("Analyze CRPD documents by article keywords across regions and countries.")

colA, colB, colC, colD = st.columns(4)
with colA:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rows Analyzed", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
with colB:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Mentions", int(df.get("Mentions", pd.Series(0, index=df.index)).sum()))
        st.markdown("</div>", unsafe_allow_html=True)
with colC:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Sentiment", f"{df.get("Sentiment", pd.Series(0, index=df.index)).mean():.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
with colD:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Countries", df.get("country", pd.Series(dtype=str)).nunique())
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Overview", "Regional Analysis", "Country Explorer"])

# ---------- Overview ----------
with tab1:
    st.markdown("<h3 class='section-title'>Global Word Cloud</h3>", unsafe_allow_html=True)
    global_text = " ".join(df.get("text", pd.Series(dtype=str)).astype(str).tolist())
    fig_wc = make_wordcloud(global_text, width=1200, height=520)
    if fig_wc:
        st.pyplot(fig_wc, use_container_width=True)
    st.markdown(
        f"<p class='caption'>Most frequent terms for keywords: <b>{', '.join(keywords)}</b>. Cleaned and combined from all matching documents.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<h3 class='section-title'>Text Length Distribution</h3>", unsafe_allow_html=True)
    if "text" in df.columns:
        lengths = df["text"].astype(str).str.len()
        fig = px.histogram(lengths, nbins=40, labels={"value": "Text length (characters)"}, title="Distribution of Text Lengths")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Regional Analysis ----------
with tab2:
    st.markdown("<h3 class='section-title'>Mentions by Region</h3>", unsafe_allow_html=True)
    region_summary = (
        df.groupby("Region", dropna=False)["Mentions"].sum().reset_index().sort_values("Mentions", ascending=False)
    )

    fig_r = px.bar(
        region_summary,
        x="Region",
        y="Mentions",
        title="Total Mentions by Region",
        text="Mentions",
    )
    fig_r.update_traces(textposition="outside")
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("<h3 class='section-title'>Average Sentiment by Region</h3>", unsafe_allow_html=True)
    if "Sentiment" in df.columns:
        region_sent = df.groupby("Region", dropna=False)["Sentiment"].mean().reset_index()
        fig_s = px.bar(region_sent, x="Region", y="Sentiment", title="Average Sentiment (−1 to +1)")
        st.plotly_chart(fig_s, use_container_width=True)

    # Export region summary
    st.subheader("Export Region Summary")
    csv_buf = io.StringIO()
    region_summary.to_csv(csv_buf, index=False)
    st.download_button(
        "Download Region Mentions CSV",
        data=csv_buf.getvalue(),
        file_name="region_mentions.csv",
        mime="text/csv",
    )

# ---------- Country Explorer ----------
with tab3:
    st.markdown("<h3 class='section-title'>Country Explorer</h3>", unsafe_allow_html=True)
    countries = sorted(df.get("country", pd.Series(dtype=str)).dropna().unique().tolist())
    sel_country = st.selectbox("Select a country", countries)

    ctry_df = df[df.get("country", "") == sel_country]
    st.write(f"**Rows for {sel_country}:** {len(ctry_df):,}")

    # Country word cloud
    ctry_text = " ".join(ctry_df.get("text", pd.Series(dtype=str)).astype(str).tolist())
    fig_ct = make_wordcloud(ctry_text, width=1000, height=400)
    if fig_ct:
        st.pyplot(fig_ct, use_container_width=True)

    # Country sentiment + mentions
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Docs", len(ctry_df))
    with c2:
        st.metric("Mentions", int(ctry_df.get("Mentions", pd.Series(0, index=ctry_df.index)).sum()))
    with c3:
        st.metric("Avg Sentiment", f"{ctry_df.get("Sentiment", pd.Series(0, index=ctry_df.index)).mean():.3f}")

    # Show a quick table of top mention rows
    st.markdown("**Top Rows by Mentions**")
    if "Mentions" in ctry_df.columns:
        top_rows = ctry_df.sort_values("Mentions", ascending=False).head(10)
        preview_cols = [c for c in ["doc_type", "country", "Region", "Mentions", "Sentiment", "text"] if c in top_rows.columns]
        st.dataframe(top_rows[preview_cols], use_container_width=True)

    # Download filtered data
    st.subheader("Export Country Rows")
    cbuf = io.StringIO()
    ctry_df.to_csv(cbuf, index=False)
    st.download_button(
        f"Download {sel_country} rows as CSV",
        data=cbuf.getvalue(),
        file_name=f"{sel_country.replace(' ', '_').lower()}_rows.csv",
        mime="text/csv",
    )

# =========================
# Footer
# =========================
st.markdown("""
---
*CRPD Article Explorer • Streamlit • v2*
""")
