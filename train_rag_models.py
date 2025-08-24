import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def train_rag_models():
    # Load QA pairs
    with open('qa_pairs.json', 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    qa_texts = [f"Q: {pair['Q']}\nA: {pair['A']}" for pair in qa_pairs]

    # Chunking (if needed)
    def chunk_text(text, chunk_size, tokenizer):
        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk)
        return chunks

    from transformers import AutoTokenizer
    chunk_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    chunks_100 = []
    for text in qa_texts:
        chunks_100.extend(chunk_text(text, 100, chunk_tokenizer))
    chunk_texts = chunks_100

    # Dense Embedding Model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
    np.save('rag_embeddings.npy', embeddings)
    embedding_model.save('rag_embedding_model')

    # FAISS Index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, 'rag_faiss.index')

    # Sparse Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    import pickle
    with open('rag_tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('rag_tfidf_matrix.npy', 'wb') as f:
        np.save(f, tfidf_matrix.toarray())

    # Save chunk texts
    with open('rag_chunk_texts.json', 'w', encoding='utf-8') as f:
        json.dump(chunk_texts, f)

    print("RAG models and data saved: rag_embeddings.npy, rag_embedding_model, rag_faiss.index, rag_tfidf_vectorizer.pkl, rag_tfidf_matrix.npy, rag_chunk_texts.json")
