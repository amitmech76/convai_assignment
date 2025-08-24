import streamlit as st
import traceback

st.set_page_config(page_title="Financial QA Setup", layout="wide")

st.title("Financial QA App Setup")

st.write("Use the buttons below to run heavy setup tasks on demand. This avoids running long jobs at import time and prevents Streamlit context warnings.")

col1, col2 = st.columns(2)

with col1:
    if st.button('Run fine-tuning (finetune_distilgpt2)'):
        try:
            from finetune_distilgpt2 import finetune_distilgpt2
            with st.spinner('Running fine-tuning... this may take a long time'):
                finetune_distilgpt2()
            st.success('Fine-tuning finished (or started). Check logs above.')
        except Exception:
            st.error('Error running fine-tuning script — see details')
            st.exception(traceback.format_exc())

    if st.button('Run RAG training (train_rag_models)'):
        try:
            from train_rag_models import train_rag_models
            with st.spinner('Training RAG models... this may take a long time'):
                train_rag_models()
            st.success('RAG training finished. Artifacts saved to disk.')
        except Exception:
            st.error('Error running RAG training — see details')
            st.exception(traceback.format_exc())

with col2:
    st.write('Launch the QA UI below. If model artifacts are missing, you will see helpful errors.')
    if st.button('Launch Financial QA UI'):
        try:
            from financial_qa_app import run_financial_qa_app
            run_financial_qa_app()
        except Exception:
            st.error('Error launching Financial QA UI — see details')
            st.exception(traceback.format_exc())

st.write('Notes:')
st.markdown('- Avoid running this app locally unless you have the required packages installed (`pip install -r requirements.txt`).')
st.markdown('- On Streamlit Community Cloud, ensure `requirements.txt` is present. Do not run heavy training there; prefer a separate VM for training.')
