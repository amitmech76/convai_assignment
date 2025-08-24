import streamlit as st
import subprocess

st.title("Financial QA App Setup")

st.write("Running finetune_distilgpt2.py...")
finetune_result = subprocess.run(["python", "finetune_distilgpt2.py"], capture_output=True, text=True)
st.text(finetune_result.stdout)

st.write("Running train_rag_models.py...")
train_result = subprocess.run(["python", "train_rag_models.py"], capture_output=True, text=True)
st.text(train_result.stdout)

st.write("Launching Financial QA App...")
# Import and run your main app logic here
try:
    import financial_qa_app
except Exception as e:
    st.error(f"Error running financial_qa_app.py: {e}")
