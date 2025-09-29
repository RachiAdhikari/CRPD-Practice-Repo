import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import altair as alt

# ======================
# Load Data
# ======================
df = pd.read_csv("CRPD_reports.csv")

# Filter for Concluding Observations mentioning "justice" (Article 13)
article13 = df[(df['doc_type']=="concluding observations") & (df['text'].str.contains("justice", case=False))]

# Make a copy to avoid SettingWithCopyWarning
article13 = article13.copy()

# Map countries to regions (expand dictionary as needed)
region_map = {
    "Kenya":"Africa", "South Africa":"Africa",
    "Nepal":"Asia-Pacific", "India":"Asia-Pacific",
    "Brazil":"Latin America", "Chile":"Latin America",
    "Germany":"Europe", "France":"Europe",
    "Canada":"North America", "United States":"North America"
}
article13.loc[:, 'Region'] = article13['country'].map(region_map)

# ======================
# Dashboard Title
# ======================
st.title("Access to Justice in CRPD Reports (Article 13)")

st.markdown("""
This dashboard highlights how **Article 13 (Access to Justice)** is addressed in Concluding Observations.
We examine how often justice is mentioned across regions, and the overall tone of these observations.
""")

# ======================
# Global Word Cloud
# ======================
st.header("Global Mentions of Justice")
text = " ".join(article13['text'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

st.caption("This word cloud shows the most common terms related to **justice** across all Concluding Observations globally.")

# ======================
# Regional Mentions Bar Chart
# ======================
st.header("Regional Focus on Access to Justice")

# Count justice mentions case-insensitively
article13['Justice Mentions'] = article13['text'].str.lower().str.count("justice")
region_counts = article13.groupby("Region")['Justice Mentions'].sum().reset_index()

chart = alt.Chart(region_counts).mark_bar(color="#4BA3C3").encode(
    x=alt.X('Region:N', title='Region'),
    y=alt.Y('Justice Mentions:Q', title='Mentions of Justice'),
    tooltip=['Region','Justice Mentions']
).properties(width=650, height=400)

st.altair_chart(chart)

st.caption("""
This chart shows how often **justice-related issues** are explicitly flagged in Concluding Observations by region. 
More mentions suggest greater attention from the Committee — often reflecting **concerns about justice accessibility** in that region.
""")

# ======================
# Sentiment Analysis
# ======================
st.header("Tone of Concluding Observations by Region")

article13['Sentiment'] = article13['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
region_sentiment = article13.groupby("Region")['Sentiment'].mean().reset_index()

tone_chart = alt.Chart(region_sentiment).mark_bar(color="#80CFA9").encode(
    x=alt.X('Region:N', title='Region'),
    y=alt.Y(
        'Sentiment:Q',
        title='Sentiment Score (-1 = Negative, +1 = Positive)',
        axis=alt.Axis(labelAngle=0, titlePadding=25, labelAlign="right")
    ),
    tooltip=['Region','Sentiment']
).properties(width=650, height=400)

st.altair_chart(tone_chart)

st.caption("""
Across all regions, the tone of Concluding Observations is generally **neutral** (scores near 0).  
Asia-Pacific and Latin America include slightly more **positive wording** (e.g., *progress noted, welcomes*),  
while Europe and North America lean more **critical** (e.g., *concerns, barriers*).  
""")


# ======================
# Country Explorer
# ======================
st.header("Country Explorer: Access to Justice")

countries = ["Kenya","Nepal","Brazil","Germany","Canada","South Africa"]
country = st.selectbox("Select a country", countries)

country_text = " ".join(article13[article13['country']==country]['text'])

wc = WordCloud(width=600, height=300, background_color="white").generate(country_text)
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.imshow(wc, interpolation="bilinear")
ax2.axis("off")
st.pyplot(fig2)

sent = TextBlob(country_text).sentiment.polarity
st.write(f"**Sentiment Score for {country}:** {round(sent,3)}")

st.caption("The word cloud and sentiment score provide a snapshot of how **Access to Justice** is framed in this country’s Concluding Observations.")
