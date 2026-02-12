import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# -------------------------------
# Step 0: Download required NLTK data
# -------------------------------
nltk.download('punkt')

# -------------------------------
# Step 1: Load precomputed data
# -------------------------------
with st.spinner("Loading Reuters dataset..."):
    embeddings = np.load("embeddings.npy")  # Precomputed embeddings
    with open("documents_original.txt", "r", encoding="utf-8") as f:
        documents_original = f.readlines()  # Original readable text

    model = Word2Vec.load("word2vec_reuters.model")  # Trained Word2Vec model

st.success(f"Loaded {len(documents_original)} documents from Reuters.")
st.info(f"Vocabulary size: {len(model.wv.index_to_key)}")

# -------------------------------
# Step 2: Helper functions
# -------------------------------
def get_query_embedding(query):
    tokens = [w.lower() for w in word_tokenize(query) if w.isalnum() and w.lower() in model.wv]
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def retrieve_top_k(query_embedding, embeddings, k=10):
    """
    Retrieve top-k most similar documents using cosine similarity.
    Returns (document_text, similarity_score)
    """
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents_original[i], similarities[i]) for i in top_k_indices]

# -------------------------------
# Step 3: Streamlit UI
# -------------------------------
st.title("🔍 Information Retrieval System")
st.subheader("Search Reuters News Articles Using Word Embeddings")

query = st.text_input("Enter your search query:")

if st.button("Search") and query.strip():
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings, k=10)
    
    # Display results
    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")