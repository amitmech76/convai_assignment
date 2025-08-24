import streamlit as st
import time
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import pickle
import traceback


def run_financial_qa_app():
    """Run the Streamlit UI for the financial QA app. All heavy IO is wrapped in try/except
    so the app can present meaningful errors instead of crashing the server."""
    st.title('Comparative Financial QA: RAG vs Fine-Tuned Model')
    st.write('Ask financial questions and compare answers from RAG and Fine-Tuned models. Guardrails are applied to input and output.')

    # Lazy-load resources and report exceptions
    @st.cache(allow_output_mutation=True)
    def load_resources():
        res = {}
        try:
            res['rag_embeddings'] = np.load('rag_embeddings.npy')
        except Exception as e:
            raise RuntimeError('Failed to load rag_embeddings.npy') from e
        try:
            res['embedding_model'] = SentenceTransformer('rag_embedding_model')
        except Exception as e:
            raise RuntimeError('Failed to load rag_embedding_model') from e
        try:
            res['index'] = faiss.read_index('rag_faiss.index')
        except Exception as e:
            raise RuntimeError('Failed to load rag_faiss.index') from e
        try:
            with open('rag_tfidf_vectorizer.pkl', 'rb') as f:
                res['vectorizer'] = pickle.load(f)
        except Exception as e:
            raise RuntimeError('Failed to load rag_tfidf_vectorizer.pkl') from e
        try:
            res['tfidf_matrix'] = np.load('rag_tfidf_matrix.npy')
        except Exception as e:
            raise RuntimeError('Failed to load rag_tfidf_matrix.npy') from e
        try:
            with open('rag_chunk_texts.json', 'r', encoding='utf-8') as f:
                res['chunk_texts'] = json.load(f)
        except Exception as e:
            raise RuntimeError('Failed to load rag_chunk_texts.json') from e

        # Fine-tuned model
        try:
            ft_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            ft_tokenizer.pad_token = ft_tokenizer.eos_token
            res['ft_tokenizer'] = ft_tokenizer
        except Exception as e:
            raise RuntimeError('Failed to load tokenizer for fine-tuned model') from e
        try:
            res['ft_model'] = AutoModelForCausalLM.from_pretrained('ft_distilgpt2_model')
        except Exception as e:
            raise RuntimeError('Failed to load ft_distilgpt2_model') from e

        return res

    # Attempt to load resources and show traceback on error
    resources = None
    try:
        resources = load_resources()
    except Exception as e:
        st.error('Error loading model resources — see details below')
        st.exception(traceback.format_exc())
        return

    # Define helper functions that use loaded resources
    def generate_ft_answer(question, max_length=64):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resources['ft_model'].to(device)
        tokenizer = resources['ft_tokenizer']
        input_ids = tokenizer.encode(question + '\nAnswer:', return_tensors='pt').to(device)
        output = model.generate(input_ids, max_length=input_ids.shape[1]+max_length, do_sample=True, num_return_sequences=1)
        answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return answer.strip()

    def hybrid_retrieve(query, top_n=5, alpha=0.5):
        embedding_model = resources['embedding_model']
        index = resources['index']
        vectorizer = resources['vectorizer']
        tfidf_matrix = resources['tfidf_matrix']
        chunk_texts = resources['chunk_texts']

        query_embedding = embedding_model.encode([query])
        D, I = index.search(query_embedding, top_n)
        dense_results = [(i, D[0][j]) for j, i in enumerate(I[0])]
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec.toarray(), tfidf_matrix).flatten()
        top_sparse_idx = np.argsort(scores)[::-1][:top_n]
        sparse_results = [(i, scores[i]) for i in top_sparse_idx]
        combined = {}
        for i, score in dense_results:
            combined[i] = alpha * score
        for i, score in sparse_results:
            combined[i] = combined.get(i, 0) + (1 - alpha) * score
        sorted_idx = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        top_chunks = [chunk_texts[i] for i, _ in sorted_idx[:top_n]]
        return top_chunks

    def multi_stage_retrieve(query, top_k=5):
        initial_chunks = hybrid_retrieve(query, top_n=top_k*2)
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, chunk] for chunk in initial_chunks]
        scores = cross_encoder.predict(pairs)
        reranked = [chunk for _, chunk in sorted(zip(scores, initial_chunks), key=lambda x: x[0], reverse=True)]
        return reranked[:top_k]

    # Guardrails
    FINANCIAL_KEYWORDS = [
        'revenue', 'profit', 'income', 'balance sheet', 'cash flow', 'assets', 'liabilities', 'equity', 'expenses', 'earnings', 'dividend', 'share', 'stock', 'financial', 'statement', 'year', 'quarter', 'company', 'net', 'gross', 'operating', 'EBITDA', 'debt', 'ratio', 'cost', 'tax', 'sales', 'turnover', 'margin', 'capital', 'investment', 'report'
    ]
    HALLUCINATION_PATTERNS = [
        r'I do not know', r'not available', r'no data', r'cannot answer', r'unknown', r'N/A', r'none'
    ]

    def validate_query(query):
        if re.search(r'(kill|hate|bomb|attack|sex|drugs)', query, re.IGNORECASE):
            return False, 'Query contains harmful or inappropriate content.'
        if not any(word in query.lower() for word in FINANCIAL_KEYWORDS):
            return False, 'Query is not related to financial data.'
        return True, ''

    def check_output_guardrail(answer):
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                return False, 'Output may be hallucinated or non-factual.'
        return True, ''

    # Streamlit interaction
    query = st.text_input('Enter your financial question:')
    method = st.radio('Select QA Method:', ['RAG (Multi-Stage Retrieval)', 'Fine-Tuned Model'])

    if st.button('Get Answer'):
        valid, msg = validate_query(query)
        if not valid:
            st.warning(f'Input Guardrail: {msg}')
        else:
            start_time = time.time()
            try:
                if method == 'RAG (Multi-Stage Retrieval)':
                    retrieved_chunks = multi_stage_retrieve(query, top_k=5)
                    # generate_answer uses generator model in original code; here we use retrieved chunks as context and FT model as fallback
                    answer = '\n'.join(retrieved_chunks)[:512]
                    method_name = 'RAG (Multi-Stage Retrieval)'
                    confidence = 'N/A'
                else:
                    answer = generate_ft_answer(query)
                    method_name = 'Fine-Tuned Model'
                    confidence = 'N/A'
            except Exception:
                st.error('Error while generating answer — see traceback')
                st.exception(traceback.format_exc())
                return
            elapsed = time.time() - start_time
            ok, out_msg = check_output_guardrail(answer)
            if not ok:
                st.warning(f'Output Guardrail: {out_msg}')
            st.text_area('Answer', value=answer, height=200)
            st.write('Method:', method_name)
            st.write('Response Time:', f'{elapsed:.2f} seconds')
            st.write('Confidence:', confidence)


if __name__ == '__main__':
    run_financial_qa_app()
