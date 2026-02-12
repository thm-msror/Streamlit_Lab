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
with st.spinner("Loading precomputed embeddings and documents..."):
    embeddings = np.load("embeddings.npy")
    model = Word2Vec.load("word2vec_reuters.model")

    # Load documents preserving multi-line content
    documents_with_id = []
    with open("documents_original.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        raw_docs = raw_text.split("====DOC====\n")

        for idx, doc in enumerate(raw_docs):
            doc = doc.strip()
            if not doc:
                continue
            lines = doc.split("\n")
            title = lines[0] if lines else "No Title"
            documents_with_id.append((f"Doc_{idx+1}", title, doc))

st.success(f"Loaded {len(documents_with_id)} documents.")
st.info(f"Vocabulary size: {len(model.wv.index_to_key)}")

# -------------------------------
# Step 2: Helper functions
# -------------------------------
def get_query_embedding(query):
    tokens = [w.lower() for w in word_tokenize(query) if w.isalnum()]
    vectors = [model.wv[w] for w in tokens if w in model.wv]

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]

    results = []
    for i in top_k_indices:
        score = similarities[i]
        if score > 0:
            fid, title, content = documents_with_id[i]
            results.append((fid, title, content, score))
    return results

# -------------------------------
# Step 3: Streamlit UI
# -------------------------------
st.title("🔍 Information Retrieval System")
st.subheader("Search Reuters News Articles Using Word Embeddings")

query = st.text_input("Enter your search query:")

if st.button("Search") and query.strip():
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings, k=10)

    num_results = len(results)

    if num_results == 0:
        st.warning("No relevant documents found for this query.")
    else:
        st.success(f"{num_results} document(s) retrieved.")
        st.write(f"### Showing Top {num_results} Relevant Documents:")

        for fid, title, content, score in results:
            st.write(f"**FieldID:** {fid}")
            st.write(f"**Title:** {title}")
            st.write(f"**Content:** {content[:500]}...")
            st.write(f"**Score:** {score:.4f}")
            st.markdown("---")