# Streamlit_Lab

Streamlit is a powerful framework for creating web applications with Python. This lab activity will, you will go through the process of hosting an Information Retrieval (IR) app using document embeddings on Streamlit. The app will allow users to enter a query and retrieve the top K most relevant documents.

## Features

* Search through Reuters news articles using semantic similarity.
* Retrieve top-K relevant documents using **Word2Vec embeddings** and **cosine similarity**.
* Handles query capitalization (e.g., `US` or `us`) automatically.
* Displays:

  * Field ID (Reuters fileid)
  * Document title (first few words of content)
  * Document content preview
  * Similarity score
* Filters out irrelevant results (documents with similarity score = 0).

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Streamlit_Lab.git
cd Streamlit_Lab
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Ensure the following files are present:

* `app.py` – the main Streamlit app
* `embeddings.npy` – precomputed document embeddings
* `documents_original.txt` – Reuters documents
* `word2vec_reuters.model` – trained Word2Vec model

---

## Usage

Run the Streamlit application locally:

```bash
streamlit run app.py
```

Or, if you get an error:

```bash
python -m streamlit run app.py
```

Open your browser at: [http://localhost:8501](http://localhost:8501)

Enter a query in the search box and click **Search** to view the top relevant documents.

---

## Project Structure

```
Streamlit_Lab/
│
├── app.py                     # Streamlit app
├── embeddings.npy             # Precomputed Word2Vec document embeddings
├── documents_original.txt     # Original Reuters documents
├── word2vec_reuters.model     # Trained Word2Vec model
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

---

## Notes

* The app currently uses **Word2Vec embeddings** for semantic similarity.
* Document titles are extracted as the first few words of the document.
* Only documents with a **positive similarity score** are shown in results.

---