import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------
# Page configuration
# ------------------------------------------------
st.set_page_config(
    page_title="UMPSA Learning Recommendation System",
    layout="wide"
)

st.title("🎓 UMPSA Learning Recommendation System")
st.write(
    "This system recommends learning content based on topic similarity "
    "using a content-based recommendation approach."
)

# ------------------------------------------------
# Load dataset
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("umpsa_fyp_data.xlsx")

    # Ensure text columns are strings
    for col in [
        "title", "Category", "keywords",
        "description", "difficulty", "Type"
    ]:
        df[col] = df[col].astype(str)

    # Text representation for TF-IDF
    df["text"] = (
        df["title"] + " " +
        df["Category"] + " " +
        df["keywords"] + " " +
        df["description"]
    )

    return df

data = load_data()

# ------------------------------------------------
# TF-IDF Model
# ------------------------------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(data["text"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# ------------------------------------------------
# Sidebar – optional user filters
# ------------------------------------------------
st.sidebar.header("Optional Preferences")

faculty = st.sidebar.selectbox(
    "Faculty",
    ["All"] + sorted(data["Faculty"].unique())
)

category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(data["Category"].unique())
)

difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["All"] + sorted(data["difficulty"].unique())
)

# ------------------------------------------------
# Main Interface
# ------------------------------------------------
st.subheader("🔍 Select a Topic")

selected_title = st.selectbox(
    "Choose a topic you are interested in:",
    data["title"].unique()
)

selected_index = data[data["title"] == selected_title].index[0]

# ------------------------------------------------
# Recommendation Logic
# ------------------------------------------------
if st.button("✨ Recommend Similar Content"):

    results = data.copy()

    # Similarity score
    results["similarity"] = similarity_matrix[selected_index]

    # Sort by similarity (most relevant first)
    recommendations = results.sort_values(
        "similarity", ascending=False
    )

    # Remove the selected item itself
    recommendations = recommendations[
        recommendations["title"] != selected_title
    ]

    # Remove duplicate titles (IMPORTANT FIX)
    recommendations = recommendations.drop_duplicates(
        subset=["title"]
    )

    # Apply OPTIONAL filters (simple & safe)
    if faculty != "All":
        recommendations = recommendations[
            recommendations["Faculty"] == faculty
        ]

    if category != "All":
        recommendations = recommendations[
            recommendations["Category"] == category
        ]

    if difficulty != "All":
        recommendations = recommendations[
            recommendations["difficulty"] == difficulty
        ]

    # Take top 5
    recommendations = recommendations.head(5)

    # ------------------------------------------------
    # Display Results
    # ------------------------------------------------
    st.subheader("📚 Recommended Learning Content")

    if recommendations.empty:
        st.info(
            "No recommendations found with the selected preferences. "
            "Try relaxing the filters."
        )
    else:
        for _, row in recommendations.iterrows():
            with st.container(border=True):

                st.markdown(f"### {row['title']}")
                st.write(f"**Faculty:** {row['Faculty']}")
                st.write(f"**Category:** {row['Category']}")
                st.write(f"**Difficulty:** {row['difficulty']}")
                st.write(f"**Type:** {row['Type']}")

                st.caption(
                    "Recommended based on similarity in topic keywords "
                    "and content description."
                )

                st.markdown(
                    f"🔗 [Learn More]({row['resource_url']})",
                    unsafe_allow_html=True
                )

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.caption(
    "UMPSA | Content-Based Learning Recommendation System"
)