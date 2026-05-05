import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ✅ MUST BE FIRST
st.set_page_config(page_title="AI Insight Dashboard", layout="wide")

import matplotlib.pyplot as plt
import numpy as np

from backend.data_processing import load_data, clean_text, get_sentiment
from backend.embedding import load_embedding_model, create_embeddings
from backend.vector_store import create_faiss_index
from backend.rag_engine import load_llm, generate_insight


# ==============================
# 🚀 LOAD SYSTEM (CACHE)
# ==============================
@st.cache_resource
def load_system():
    df = load_data("Reviews.csv")
    df = df.head(500)

    df['sentiment'] = df['Score'].apply(get_sentiment)
    df['clean_text'] = df['Text'].apply(clean_text)

    embedding_model = load_embedding_model()

    if os.path.exists("embeddings.npy"):
        embeddings = np.load("embeddings.npy")
    else:
        embeddings = create_embeddings(embedding_model, df['clean_text'].tolist())
        np.save("embeddings.npy", embeddings)

    index = create_faiss_index(embeddings)
    generator = load_llm()

    return df, embedding_model, index, generator


df, embedding_model, index, generator = load_system()


# ==============================
# 🎯 SIDEBAR (PRODUCT STYLE)
# ==============================
st.sidebar.title("⚙️ Controls")

query = st.sidebar.text_input("🔍 Enter your query")

st.sidebar.markdown("### 💡 Quick Queries")

if st.sidebar.button("🚚 Delivery Issues"):
    query = "Why are customers unhappy with delivery?"

if st.sidebar.button("📦 Packaging Problems"):
    query = "What issues are related to packaging?"

if st.sidebar.button("📞 Service Complaints"):
    query = "What problems exist in customer service?"


# ==============================
# 🧠 HEADER
# ==============================
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🧠 AI Customer Insight Engine
</h1>
<p style='text-align: center; color: grey; font-size:18px;'>
Turn raw reviews into actionable insights using AI
</p>
""", unsafe_allow_html=True)


# ==============================
# 🔥 MAIN OUTPUT
# ==============================
if query:
    with st.spinner("🔍 Analyzing customer feedback..."):
        insight, reviews = generate_insight(
            query,
            embedding_model,
            index,
            df,
            generator
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    # 📄 REVIEWS (CARD STYLE)
    with col1:
        st.subheader("📄 Retrieved Reviews")

        for r in reviews:
            st.markdown(f"""
            <div style="
                background-color:#262730;
                padding:12px;
                border-radius:10px;
                margin-bottom:10px;
                color:white;
                font-size:14px;
            ">
            {r}
            </div>
            """, unsafe_allow_html=True)

    # 💡 INSIGHT (HIGHLIGHT BOX)
    with col2:
        st.subheader("💡 AI Insight")

        st.markdown(f"""
        <div style="
            background-color:#0e1117;
            padding:20px;
            border-radius:10px;
            border-left:5px solid #4CAF50;
            color:white;
            font-size:16px;
        ">
        {insight.replace('\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)


# ==============================
# 📊 SENTIMENT CHART
# ==============================
st.markdown("---")
st.markdown("### 📊 Sentiment Overview")

sentiment_counts = df['sentiment'].value_counts()

fig, ax = plt.subplots()

ax.bar(sentiment_counts.index, sentiment_counts.values)

for i, v in enumerate(sentiment_counts.values):
    ax.text(i, v + 1, str(v), ha='center')

ax.set_title("Customer Sentiment Distribution")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

st.pyplot(fig)