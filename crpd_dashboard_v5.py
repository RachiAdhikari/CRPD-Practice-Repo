# =====================================================
# 🌍 CRPD Disability Rights Data Dashboard (v5)
# -----------------------------------------------------
# - Uses CRPD_dashboard_ready.csv
# - Uses crpd_article_dict.py (article keyword mapping)
# - Dynamic TF-IDF thresholds (no ValueError on small subsets)
# - Tabs: Overview, CRPD Articles, Keywords & Topics, Comparative, Country Explorer
# =====================================================


import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="CRPD Disabililty Rights Data Dashboard",
    page_icon="🌍",
    layout="wide",
)

HIDE_SIDEBAR_STYLE = """
    <style>
      .block-container{padding-top:1.2rem;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
"""
st.markdown(HIDE_SIDEBAR_STYLE, unsafe_allow_html=True)

# -------------------------
# Load Data & Dictionaries
# -------------------------
@st.cache_data(show_spinner=True)
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ["doc_type", "country", "region", "subregion", "language"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "text_snippet" not in df.columns and "clean_text" in df.columns:
        df["text_snippet"] = df["clean_text"].apply(lambda x: " ".join(str(x).split()[:120]))
    return df

@st.cache_data
def load_article_dict():
    try:
        from crpd_article_dict import ARTICLE_PRESETS
        return ARTICLE_PRESETS
    except Exception as e:
        st.warning(f"Couldn't load article dictionary ({e}); using fallback.")
        return {
            "Article 9 — Accessibility": ["accessibility", "barrier", "universal design"],
            "Article 13 — Access to Justice": ["justice", "court", "legal"],
            "Article 24 — Education": ["education", "school", "inclusive education"]
        }

MODEL_DICT = {
    "Medical Model": [
        "treatment","rehabilitation","therapy","patient","disorder","impairment",
        "illness","diagnosis","caregiver","institution","special needs","cure"
    ],
    "Rights-Based Model": [
        "inclusion","equality","accessibility","participation","autonomy",
        "independent living","reasonable accommodation","universal design",
        "dignity","rights","empowerment","access to justice"
    ]
}

# -------------------------
# Helper Functions
# -------------------------
def filter_df(df, region, country, doc_types, year_range):
    d = df.copy()
    if region and region != "All":
        d = d[d["region"] == region]
    if country and country != "All":
        d = d[d["country"] == country]
    if doc_types:
        d = d[d["doc_type"].isin(doc_types)]
    if year_range and "year" in d.columns:
        ymin, ymax = year_range
        d = d[(d["year"].fillna(0) >= ymin) & (d["year"].fillna(9999) <= ymax)]
    return d

def count_phrases(text, phrases):
    if not isinstance(text, str):
        return 0
    total = 0
    for kw in phrases:
        total += len(re.findall(r"\b" + re.escape(kw) + r"\b", text))
    return total

@st.cache_data
def article_frequency(df, article_dict, groupby=None):
    rows = []
    iterable = [(None, df)] if not groupby else df.groupby(groupby)
    for g, sub in iterable:
        for art, kws in article_dict.items():
            c = sub["clean_text"].apply(lambda t: count_phrases(t, kws)).sum()
            rows.append({"group": ("All" if g is None else g), "article": art, "count": int(c)})
    out = pd.DataFrame(rows)
    return out[out["count"] > 0].sort_values("count", ascending=False)

@st.cache_data
def keyword_counts(df, top_n=30):
    cnt = Counter()
    for t in df["clean_text"].astype(str).tolist():
        cnt.update(w for w in t.split() if 2 <= len(w) <= 25)
    return pd.DataFrame(cnt.items(), columns=["term", "freq"]).sort_values("freq", ascending=False).head(top_n)

@st.cache_data
def tfidf_by_doc_type(df, top_n=20):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        st.warning("scikit-learn not installed; using frequency fallback.")
        return keyword_counts(df, top_n).assign(doc_type="All").rename(columns={"freq":"score"})
    rows = []
    for dt, sub in df.groupby("doc_type"):
        docs = sub["clean_text"].dropna().astype(str).tolist()
        if len(docs) < 2:
            topk = keyword_counts(sub, top_n)
            topk["doc_type"] = dt
            rows.append(topk.rename(columns={"freq": "score"}))
            continue
        n_docs = len(docs)
        min_df = 1 if n_docs < 10 else 2
        max_df = 1.0 if n_docs <= 3 else 0.9
        try:
            vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, 2))
            mat = vec.fit_transform(docs)
            terms = np.array(vec.get_feature_names_out())
            scores = np.asarray(mat.mean(axis=0)).ravel()
            idx = scores.argsort()[::-1][:top_n]
            tmp = pd.DataFrame({"term": terms[idx], "score": scores[idx], "doc_type": dt})
            rows.append(tmp)
        except ValueError:
            topk = keyword_counts(sub, top_n)
            topk["doc_type"] = dt
            rows.append(topk.rename(columns={"freq": "score"}))
    return pd.concat(rows, ignore_index=True)

@st.cache_data
def model_shift_table(df):
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("clean_text", ""))
        counts = {m: count_phrases(text, kws) for m, kws in MODEL_DICT.items()}
        total = sum(counts.values()) if sum(counts.values()) > 0 else 1
        rows.append({
            "region": r.get("region","Unknown"),
            "year": r.get("year", np.nan),
            "medical": counts["Medical Model"],
            "rights": counts["Rights-Based Model"],
            "rights_share": counts["Rights-Based Model"]/total
        })
    return pd.DataFrame(rows)

# -------------------------
# Load Data
# -------------------------
DATA_PATH = "CRPD_dashboard_ready.csv"
df_all = load_data(DATA_PATH)
ARTICLE_PRESETS = load_article_dict()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.markdown("### 🔍 Filters")

regions = ["All"] + sorted(df_all["region"].dropna().unique())
region = st.sidebar.selectbox("Region", regions, index=0)

countries = ["All"] + sorted(df_all.loc[(df_all["region"] == region) | (region == "All"), "country"].unique())
country = st.sidebar.selectbox("Country", countries, index=0)

doc_types_all = sorted(df_all["doc_type"].unique())
doc_types = st.sidebar.multiselect("Document Type", doc_types_all, default=doc_types_all)

ymin, ymax = int(df_all["year"].min()), int(df_all["year"].max())
year_range = st.sidebar.slider("Year Range", ymin, ymax, (ymin, ymax))

download_toggle = st.sidebar.checkbox("Include text snippets in export", value=False)

df = filter_df(df_all, region, country, doc_types, year_range)

# -------------------------
# Header & KPIs
# -------------------------
st.title("🌍 CRPD Disability Rights Data Dashboard")
st.caption("Explore how countries implement the UN Convention on the Rights of Persons with Disabilities (CRPD) across all report types.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Documents", f"{len(df):,}")
k2.metric("Countries", f"{df['country'].nunique():,}")
k3.metric("Regions", f"{df['region'].nunique():,}")
if "year" in df.columns and len(df):
    k4.metric("Years", f"{int(df['year'].min())}–{int(df['year'].max())}")
else:
    k4.metric("Years", "—")

# -------------------------
# Tabs
# -------------------------
tab_overview, tab_articles, tab_keywords, tab_compare, tab_country = st.tabs(
    ["Overview", "CRPD Articles", "Keywords & Topics", "Comparative", "Country Explorer"]
)

# === OVERVIEW ===
with tab_overview:
    st.subheader("Global Overview")
    st.markdown("""
    Provides a high-level view of the CRPD reporting landscape — which countries report, 
    how often, and how patterns change over time.
    """)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        counts = df.groupby("country").size().reset_index(name="documents")
        if not counts.empty:
            st.plotly_chart(px.choropleth(
                counts, locations="country", locationmode="country names",
                color="documents", color_continuous_scale="Blues"
            ), use_container_width=True)
            st.caption("🌍 Number of CRPD documents available per country.")
    with col2:
        type_counts = df.groupby("doc_type").size().reset_index(name="count")
        st.plotly_chart(px.bar(type_counts, x="doc_type", y="count"), use_container_width=True)
        st.caption("📊 Distribution of report types (State Reports, LOIs, COs, etc.).")
        yearly = df.groupby("year").size().reset_index(name="count").sort_values("year")
        st.plotly_chart(px.line(yearly, x="year", y="count", markers=True), use_container_width=True)
        st.caption("📈 Number of reports submitted each year.")

# === GLOBAL MODEL SHIFT (new)
mt_global = model_shift_table(df)
if len(mt_global):
    by_year_global = (
        mt_global.groupby("year")[["medical","rights"]]
        .sum().reset_index().sort_values("year")
    )
    st.plotly_chart(
        px.area(
            by_year_global,
            x="year",
            y=["medical","rights"],
            title="Global Shift in Disability Framing (Medical vs. Rights-Based Language)"
        ),
        use_container_width=True,
    )
    st.caption("⚖️ This area chart shows how the use of medical vs. rights-based language has evolved globally over time.")

# === CRPD ARTICLES ===
with tab_articles:
    st.subheader("CRPD Article Coverage")
    st.markdown("""
    Shows which CRPD rights (e.g., education, accessibility, justice) are most emphasized globally or by category.
    """)
    group_choice = st.selectbox("Group by", ["All", "region", "doc_type"])
    grouping = None if group_choice == "All" else group_choice
    art_df = article_frequency(df, ARTICLE_PRESETS, groupby=grouping)
    if art_df.empty:
        st.info("No article matches for selected filters.")
    else:
        if grouping:
            topN = art_df.groupby("group").head(12)
            fig = px.bar(topN, x="article", y="count", color="group", barmode="group")
        else:
            topN = art_df.groupby("article")["count"].sum().reset_index().nlargest(12,"count")
            fig = px.bar(topN, x="article", y="count")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📘 Most frequently mentioned CRPD articles.")

# === KEYWORDS ===
with tab_keywords:
    st.subheader("Keyword & Topic Exploration")
    st.markdown("Explore recurring themes and unique language across CRPD documents.")
    colL, colR = st.columns(2)
    with colL:
        freq_df = keyword_counts(df)
        st.plotly_chart(px.bar(freq_df.sort_values("freq"), x="freq", y="term", orientation="h"), use_container_width=True)
        st.caption("💬 Most frequent words in the selected dataset.")
    with colR:
        tfidf_df = tfidf_by_doc_type(df)
        st.plotly_chart(px.bar(tfidf_df, x="score", y="term", color="doc_type", orientation="h"), use_container_width=True)
        st.caption("🧠 Top TF-IDF terms unique to each document type.")

# === COMPARATIVE ===
with tab_compare:
    st.subheader("Comparative Analysis — State vs. Committee")
    st.markdown("""
    Compare how governments and the CRPD Committee emphasize different articles and themes.
    """)
    sr = df[df["doc_type"].str.contains("State", case=False, na=False)]
    co = df[df["doc_type"].str.contains("Concluding", case=False, na=False)]
    if len(sr) and len(co):
        sr_top = article_frequency(sr, ARTICLE_PRESETS).groupby("article")["count"].sum().reset_index().nlargest(10,"count")
        co_top = article_frequency(co, ARTICLE_PRESETS).groupby("article")["count"].sum().reset_index().nlargest(10,"count")
        col1, col2 = st.columns(2)
        col1.plotly_chart(px.bar(sr_top, x="article", y="count", title="State Reports"), use_container_width=True)
        col2.plotly_chart(px.bar(co_top, x="article", y="count", title="Concluding Observations"), use_container_width=True)
        st.caption("🔹 Comparing CRPD article emphasis in State vs. Committee reports.")
    else:
        st.info("Need both State Reports and Concluding Observations to compare.")

# === COUNTRY EXPLORER ===
with tab_country:
    st.subheader("Country Explorer")
    st.markdown("""
    Country-level profile with report statistics, CRPD article mentions, and model language trends.
    """)
    ctry = st.selectbox("Select Country", sorted(df["country"].unique()))
    if ctry:
        sub = df[df["country"] == ctry]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", f"{len(sub):,}")
        c2.metric("Types", sub["doc_type"].nunique())
        if "year" in sub.columns and len(sub):
            years_display = f"{int(sub['year'].min())}–{int(sub['year'].max())}"
        else:
            years_display = "—"
        c3.metric("Years", years_display)
        c4.metric("Avg Words", int(sub["word_count"].mean()) if "word_count" in sub.columns else "—")
        sub_art = article_frequency(sub, ARTICLE_PRESETS)
        if not sub_art.empty:
            topA = sub_art.groupby("article")["count"].sum().reset_index().nlargest(10,"count")
            st.plotly_chart(px.bar(topA, x="article", y="count"), use_container_width=True)
        mt = model_shift_table(sub)
        if len(mt):
            by_year = mt.groupby("year")[["medical","rights"]].sum().reset_index().sort_values("year")
            st.plotly_chart(px.area(by_year, x="year", y=["medical","rights"], title="Model Language by Year"), use_container_width=True)
        st.dataframe(sub.sort_values("year", ascending=False)[["year","doc_type","text_snippet"]].head(10), use_container_width=True)
    else:
        st.info("Select a country to view its profile.")

# -------------------------
# Export
# -------------------------
st.markdown("---")
cols_to_export = ["doc_type","country","year","region","subregion","word_count","language","symbol","file_name"]
if download_toggle:
    cols_to_export += ["text_snippet"]
export_df = df[cols_to_export].copy() if len(df) else pd.DataFrame(columns=cols_to_export)

st.download_button(
    "⬇️ Download Filtered Data (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="CRPD_filtered_export.csv",
    mime="text/csv"
)

# -------------------------
# Footer / Attribution
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        Dashboard developed by the <b>Institute on Disability and Public Policy (IDPP)</b> research team.<br>
        © 2025 American University
    </div>
    """,
    unsafe_allow_html=True
)
