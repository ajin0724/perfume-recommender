import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

# ✅ Must be first
st.set_page_config(page_title="Perfume Recommender", layout="wide")

# 🔍 Check for valid image URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# 📄 Load data
@st.cache_data
def load_data():
    df = pd.read_csv("perfume_descriptions_with_keywords.csv")
    return df

df = load_data()

# 🖼️ Title
st.title("🌸 Perfume Recommender")
st.write("Find a fragrance that suits your mood, style, and personality.")

# 🎛️ Sidebar: user input
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

# ▶️ Recommend button
if st.sidebar.button("Recommend Perfumes"):
    input_keywords = moods + tones + personality + character_pref
    if gender != "Doesn't matter":
        input_keywords.append("man" if gender == "Male" else "woman")
    input_text = ", ".join(input_keywords)

    # 🔍 TF-IDF using only keywords
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["keywords"].fillna("").astype(str))
    user_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-5:][::-1]
    top_perfumes = df.iloc[top_indices]

    # 🎯 Display results
    st.subheader("Recommended Fragrances for You")

    for _, row in top_perfumes.iterrows():
        with st.container():
            cols = st.columns([1, 2])
            with cols[0]:
                image_url = row.get("image", None)
                if pd.notna(image_url) and isinstance(image_url, str) and is_valid_url(image_url):
                    st.image(image_url, use_container_width=True)
                else:
                    st.markdown("_No image available or invalid URL_")
            with cols[1]:
                perfume_name = row.get("Name", "Unnamed Perfume")
                brand = row.get("Brand", "N/A")
                st.markdown(
                    f"### {perfume_name} <span style='font-size: 14px; color: gray;'>({brand})</span>",
                    unsafe_allow_html=True
                )

                st.write(row.get("Description", "No description available."))

                st.markdown("**Notes:**")
                st.markdown(f"- **Top**: {row.get('Top_note', 'N/A')}")
                st.markdown(f"- **Middle**: {row.get('Middle_note', 'N/A')}")
                st.markdown(f"- **Base**: {row.get('Base_note', 'N/A')}")

        st.markdown("---")
