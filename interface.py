import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

# ✅ Streamlit configuration - must come first
st.set_page_config(page_title="Perfume Recommender", layout="wide")

# 🔍 Utility to validate image URLs
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# 📄 Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("perfume_descriptions_with_keywords.csv")
    df["combined_text"] = (
        df["keywords"].fillna("") + ", " +
        df["Character_x"].fillna("") + ", " +
        df["Fragrance_Family"].fillna("")
    )
    return df

df = load_data()

# 🖼️ App title
st.title("🌸 Perfume Recommender")
st.write("Find a fragrance that suits your mood, style, and personality.")

# 🎛️ Sidebar user input
st.sidebar.header("Your Preferences")

moods = st.sidebar.multiselect(
    "Preferred moods or feelings in a fragrance",
    ["romantic", "cozy", "bold", "fresh", "gentle", "strong", "soft", "mysterious"]
)

tones = st.sidebar.multiselect(
    "Favorite scent families",
    ["Floral", "Woody", "Oriental", "Aromatic", "Fruity", "Citrus", "Spicy", "Green", "Powdery"]
)

gender = st.sidebar.radio(
    "Do you prefer gender-based recommendations?",
    ["Doesn't matter", "Male", "Female"]
)

personality = st.sidebar.multiselect(
    "Which words describe your personality?",
    ["energetic", "calm", "confident", "delicate", "unique", "heroic", "dreamy"]
)

character_pref = st.sidebar.multiselect(
    "What kind of character or impression should the perfume express?",
    ["Romantic", "Elegant", "Mysterious", "Fresh", "Casual", "Chic", "Sexy", "Natural", "Classic"]
)

# ▶️ Recommendation logic
if st.sidebar.button("Recommend Perfumes"):
    input_keywords = moods + tones + personality + character_pref
    if gender != "Doesn't matter":
        input_keywords.append("man" if gender == "Male" else "woman")
    input_text = ", ".join(input_keywords)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"].astype(str))
    user_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-5:][::-1]
    top_perfumes = df.iloc[top_indices]

    st.subheader("🎯 Recommended Fragrances for You")

    for _, row in top_perfumes.iterrows():
        with st.container():
            cols = st.columns([1, 2])
            with cols[0]:
                image_url = row.get("image", None)
                if pd.notna(image_url) and isinstance(image_url, str) and is_valid_url(image_url):
                    st.image(image_url, use_column_width=True)
                else:
                    st.markdown("_No image available or invalid URL_")
            with cols[1]:
                st.markdown(f"### {row.get('Name', 'Unnamed Perfume')}")
                st.write(row.get("Description", "No description available."))
        st.markdown("---")
