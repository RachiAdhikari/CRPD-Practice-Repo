# crpd_dashboard_v3.py
# Streamlit dashboard for CRPD text analysis (integrated with full article dictionary)
# - Loads full article dictionary (crpd_article_dict.py)
# - Guided sidebar with Steps 1–3
# - Unified Regional & Country Explorer
# - Clean layout, contextual captions, export options
# ====================================================

import re
import io
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================
# Import Article Dictionary
# =========================
try:
    from crpd_article_dict import ARTICLE_PRESETS as BASE_PRESETS
    ARTICLE_PRESETS = {**BASE_PRESETS, "Custom": []}
except Exception as e:
    st.error(f"Could not load crpd_article_dict.py — using blank dictionary. Error: {e}")
    ARTICLE_PRESETS = {"Custom": []}

# =========================
# Page Config & Styling
# =========================
st.set_page_config(
    page_title="CRPD Article Explorer by the IDPP Team",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f7f9fc;
        border: 1px solid #e6ecf5;
        border-radius: 12px;
        padding: 14px;
    }
    h1, h2, h3, h4 { color: #0f172a; }
    .section-title {color:#155e75; border-left: 4px solid #14b8a6; padding-left:10px; margin: 8px 0 16px 0;}
    .caption {color:#475569; font-size:0.9rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background: #f1f5f9; border-radius: 8px; padding: 8px 12px;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background: #e0f2fe; border: 1px solid #38bdf8;}
</style>
""", unsafe_allow_html=True)

# =========================
# Utility Functions
# =========================
def load_data(uploaded_file: io.BytesIO | None) -> pd.DataFrame | None:
    """Load CSV from uploader or fallback to CRPD_reports.csv."""
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        return pd.read_csv("CRPD_reports.csv")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

REGION_MAP = {
    "Kenya": "Africa", "South Africa": "Africa", "Ghana": "Africa", "Nigeria": "Africa",
    "Nepal": "Asia-Pacific", "India": "Asia-Pacific", "Bangladesh": "Asia-Pacific",
    "Brazil": "Latin America", "Chile": "Latin America", "Peru": "Latin America", "Mexico": "Latin America",
    "Germany": "Europe", "France": "Europe", "Spain": "Europe", "Italy": "Europe",
    "Canada": "North America", "United States": "North America"
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
            total += text.lower().count(kw.lower())
    return total

def compute_sentiment(text: str) -> float:
    try:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0

def make_wordcloud(text: str, width: int = 1200, height: int = 600):
    if not isinstance(text, str) or not text.strip():
        return None
    cloud = WordCloud(width=width, height=height, background_color="white", max_words=150).generate(text)
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return fig

# =========================
# Sidebar — Guided Controls
# =========================
st.sidebar.header("Configuration")

# ---- STEP 1: Select Article ----
st.sidebar.markdown("### Step 1: Select Article Focus")
preset = st.sidebar.selectbox("Article", list(ARTICLE_PRESETS.keys()), index=0)
keywords = ARTICLE_PRESETS[preset]

if preset == "Custom":
    custom_kw = st.sidebar.text_input("Keywords (comma-separated)", value="justice, equality, accessibility")
    keywords = [k.strip() for k in custom_kw.split(",") if k.strip()]
else:
    st.sidebar.caption("Keywords automatically loaded for the selected article.")

# ---- STEP 2: Filter Data ----
st.sidebar.markdown("### Step 2: Filter Data")

min_len = st.sidebar.slider("Minimum text length (chars)", 0, 2000, 0, 50)

try:
    _temp = pd.read_csv("CRPD_reports.csv")
except Exception:
    _temp = pd.DataFrame(columns=["doc_type", "country", "text"])

doc_types = sorted(_temp.get("doc_type", pd.Series(dtype=str)).dropna().unique().tolist())
default_doc_types = [dt for dt in doc_types if "concluding" in str(dt).lower()] or doc_types
selected_doc_types = st.sidebar.multiselect("Document types", options=doc_types, default=default_doc_types)

all_countries = sorted(_temp.get("country", pd.Series(dtype=str)).dropna().unique().tolist())
selected_countries = st.sidebar.multiselect("Countries (optional)", options=all_countries)

# ---- STEP 3: Upload (Optional) ----
st.sidebar.markdown("---")
st.sidebar.markdown("### Step 3: Upload Your Dataset")
with st.sidebar.expander("📂 Upload CRPD CSV"):
    uploaded = st.file_uploader(
        "Drag and drop or browse CSV",
        type=["csv"],
        help="If omitted, the app loads CRPD_reports.csv from this directory."
    )

# =========================
# Load & Filter Data
# =========================
_df = load_data(uploaded)
if _df is None or _df.empty:
    st.stop()

expected_cols = {"doc_type", "text", "country"}
missing = expected_cols - set(_df.columns)
if missing:
    st.warning(f"Missing expected columns: {sorted(missing)}. Some features may not work.")

df = _df.copy()
if selected_doc_types:
    df = df[df.get("doc_type", "").isin(selected_doc_types)]
if min_len:
    df = df[df.get("text", "").astype(str).str.len() >= min_len]
if selected_countries:
    df = df[df.get("country", "").isin(selected_countries)]
if keywords:
    patt = re.compile(r"|".join([rf"\b{re.escape(k)}\b" for k in keywords]), flags=re.IGNORECASE)
    df = df[df.get("text", "").astype(str).str.contains(patt)]

if "text" in df.columns:
    df["Mentions"] = df["text"].astype(str).apply(lambda t: count_mentions(t, keywords))
    df["Sentiment"] = df["text"].astype(str).apply(compute_sentiment)

df = ensure_region_column(df)

if df.empty:
    st.info("No rows match your filters. Try broadening document types, countries, or keywords.")
    st.stop()

# =========================
# Header & KPIs
# =========================
st.title("CRPD Article Explorer")
st.caption("Analyze CRPD reports by article keywords across regions and countries.")

colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Rows Analyzed", f"{len(df):,}")
    st.markdown("</div>", unsafe_allow_html=True)
with colB:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Mentions", int(df["Mentions"].sum()))
    st.caption("Number of keyword occurrences across selected texts.")
    st.markdown("</div>", unsafe_allow_html=True)
with colC:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg Sentiment", f"{df['Sentiment'].mean():.3f}")
    st.caption("Average sentiment polarity (−1 negative, +1 positive).")
    st.markdown("</div>", unsafe_allow_html=True)
with colD:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Countries", df["country"].nunique())
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Download full filtered dataset ----
csv_all = io.StringIO()
df.to_csv(csv_all, index=False)
st.download_button("Download Filtered Dataset (CSV)", csv_all.getvalue(),
                   file_name="filtered_crpd_data.csv", mime="text/csv")

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Overview", "Regional & Country Explorer"])

# ---------- OVERVIEW ----------
with tab1:
    st.markdown("<h3 class='section-title'>Global Word Cloud</h3>", unsafe_allow_html=True)
    global_text = " ".join(df["text"].astype(str).tolist())
    fig_wc = make_wordcloud(global_text, width=1200, height=520)
    if fig_wc:
        st.pyplot(fig_wc, use_container_width=True)
    st.markdown(f"<p class='caption'>Most frequent terms for keywords: <b>{', '.join(keywords)}</b></p>", unsafe_allow_html=True)

    st.markdown("<h3 class='section-title'>Text Length Distribution</h3>", unsafe_allow_html=True)
    lengths = df["text"].astype(str).str.len()
    fig = px.histogram(lengths, nbins=40, labels={"value": "Text length (characters)"},
                       title="Distribution of Text Lengths")
    st.plotly_chart(fig, use_container_width=True)

# ---------- REGIONAL & COUNTRY ----------
with tab2:
    with st.expander("🌍 Regional Overview", expanded=True):
        st.markdown("<h3 class='section-title'>Mentions by Region</h3>", unsafe_allow_html=True)
        region_summary = df.groupby("Region", dropna=False)["Mentions"].sum().reset_index().sort_values("Mentions", ascending=False)
        fig_r = px.bar(region_summary, x="Region", y="Mentions", text="Mentions",
                       title="Total Mentions by Region")
        fig_r.update_traces(textposition="outside")
        st.plotly_chart(fig_r, use_container_width=True)

        region_sent = df.groupby("Region", dropna=False)["Sentiment"].mean().reset_index()
        fig_s = px.bar(region_sent, x="Region", y="Sentiment", title="Average Sentiment (−1 to +1)",
                       color="Sentiment", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("Regional sentiment helps assess tone and emphasis across geographic contexts.")

    with st.expander("🏳 Country Explorer", expanded=False):
        countries = sorted(df["country"].dropna().unique().tolist())
        sel_country = st.selectbox("Select a country", countries)
        ctry_df = df[df["country"] == sel_country]
        st.write(f"**Rows for {sel_country}:** {len(ctry_df):,}")

        ctry_text = " ".join(ctry_df["text"].astype(str).tolist())
        fig_ct = make_wordcloud(ctry_text, width=1000, height=400)
        if fig_ct:
            st.pyplot(fig_ct, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Docs", len(ctry_df))
        with c2:
            st.metric("Mentions", int(ctry_df["Mentions"].sum()))
        with c3:
            st.metric("Avg Sentiment", f"{ctry_df['Sentiment'].mean():.3f}")

        top_rows = ctry_df.sort_values("Mentions", ascending=False).head(10)
        preview_cols = [c for c in ["doc_type", "country", "Region", "Mentions", "Sentiment", "text"] if c in top_rows.columns]
        st.dataframe(top_rows[preview_cols], use_container_width=True)

        cbuf = io.StringIO()
        ctry_df.to_csv(cbuf, index=False)
        st.download_button(f"Download {sel_country} Rows (CSV)", cbuf.getvalue(),
                           file_name=f"{sel_country.lower()}_rows.csv", mime="text/csv")

# =========================
# Footer
# =========================
st.markdown("""
---
*CRPD Article Explorer*  
*Guided workflow, unified explorer, contextual help, and improved visuals.*
""")
