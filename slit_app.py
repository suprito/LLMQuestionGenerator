import streamlit as st
from app import generate_mcq, load_llama, downlode_hugging_face_embeddings
from langchain_chroma import Chroma


st.title("GenAI based MCQ Quiz App")

if "questions" not in st.session_state:
    st.session_state.questions = []

@st.cache_resource
def init_retriever():
    embeddings = downlode_hugging_face_embeddings()
    vectorstore = Chroma(persist_directory='chroma_db_MCQGen', embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 10})

retriever = init_retriever()

topic = st.text_input("Enter Topic:")
if st.button("Generate MCQs"):
    with st.spinner("Generating..."):
    # return dictionaries instead of raw strings.
        st.session_state.questions = generate_mcq(topic, retriever)
    st.rerun()

submit_btn = False

with st.form("quiz_form", border=False):
    for i,mcq in enumerate(st.session_state.questions):
        st.write(f"## Question {i+1}")
        st.write(f"Question: {mcq['question']}")

        user_choice=st.radio(label="Select an option", options=mcq['option'], index=None, key=f"radio_{i}")

        if st.session_state.get(f"radio_{i}") and st.session_state.get("FormSubmitter:quiz_form-Submit"):
            if user_choice.strip().lower() == mcq['correct_answer'].strip().lower(): # COMPARISON, ignore case and whitespace
                st.success("Correct Answer!", icon="✅")
            else:
                st.error("Incorrect!", icon="❌")

        with st.expander(label="Check Answer"):
            st.write(f"Correct Answer: {mcq['correct_answer']}")

    st.write("---")

    submit_btn=st.form_submit_button("Submit", type="primary")

if submit_btn:
    st.success("Quiz Submitted!")