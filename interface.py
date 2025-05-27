import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("perfume_descriptions_with_keywords.csv")

# Page title
st.title("Perfume Recommendation System")

# User input section
st.header("Tell us about your preferences!")

# Question 1: Mood
moods = st.multiselect(
    "What moods or feelings do you prefer in a fragrance?",
    ["romantic", "cozy", "bold", "fresh", "gentle", "strong", "soft", "mysterious"]
)

# Question 2: Scent family
tones = st.multiselect(
    "Which scent families do you like?",
    ["woody", "floral", "oriental", "fresh", "citrus", "spicy", "sweet", "green"]
)

# Question 3: Gender preference
gender = st.radio(
    "Would you like recommendations based on gender?",
    ["Doesn't matter", "Male", "Female"]
)

# Question 4: Personality
personality = st.multiselect(
    "Which words best describe your personality?",
    ["energetic", "calm", "confident", "delicate", "unique", "heroic", "dreamy"]
)

# Recommend button
if st.button("Get Recommendations"):
    # Combine user inputs into a single keyword string
    input_keywords = moods + tones + personality
    if gender != "Doesn't matter":
        input_keywords.append("man" if gender == "Male" else "woman")
    input_text = ", ".join(input_keywords)

    # Compute TF-IDF similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["keywords"])
    user_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    # Get top 5 matches
    top_indices = similarities.argsort()[-5:][::-1]
    top_perfumes = df.iloc[top_indices][["Name", "Description", "keywords"]]

    # Display results
    st.subheader("🎯 Recommended Perfumes")
    for _, row in top_perfumes.iterrows():
        st.markdown(f"**{row['Name']}**")
        st.write(row["Description"])
        st.caption(f"Keywords: {row['keywords']}")
