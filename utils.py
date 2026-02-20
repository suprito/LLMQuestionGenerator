import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
file_path = "data/MedicalBook.pdf"

def load_docs(file_path):
    loader=PyPDFLoader(file_path)
    documents=loader.load()
    #print(f"Document Loaded with page Numbers: {len(documents)}")
    return documents

def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200, separators=["\n\n", "\n", " ", ""],)
    text_chunks= text_splitter.split_documents(documents)
    #print(f"Total number of final chunks: {len(text_chunks)}")
    return text_chunks

def downlode_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/msmarco-MiniLM-L6-v3')
    return embeddings


CHROMA_PATH = "chroma_db_MCQGen"
def create_vectorstore(text_chunks, embeddings, persist_dir=None):
    if persist_dir:
        vectorstore=Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=CHROMA_PATH)     
    else: #in-memory version for cloud deployment
        vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings)

    return vectorstore


def load_llama():
    repo_id='meta-llama/Llama-3.2-3B-Instruct'
    llm=HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=256,
        temperature=0.3,
        repetition_penalty=1.1,
        huggingfacehub_api_token=HF_TOKEN
    )

    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

llm=load_llama()

mcq_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict medical educator. 
1. Generate ONE MCQ based ONLY on the provided context. 
2. If the context does not contain enough specific information to create a high-quality question, respond with "INSUFFICIENT_CONTEXT".
3. Ensure distractors are medically plausible but factually incorrect based on the text.
4. DO NOT provide any introductory text or conversational filler. Start immediately with 'Question:'.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Context: {context}

Follow this EXACT format:
Question: [Text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct Answer: [Letter]
Explanation: [Concise reason]<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Question:"""

mcq_prompt = ChatPromptTemplate.from_template(mcq_prompt_template)

# def generate_mcq(topic: str, retriever, num_questions: int=5): 
#     docs = retriever.invoke(topic)
#     mcq_chain = mcq_prompt | llm | StrOutputParser()
#     generated_mcqs = []
#     for i in range(min(num_questions, len(docs))):
#         context_chunk = docs[i].page_content
#         response = mcq_chain.invoke({"context": context_chunk})
#         generated_mcqs.append(response)

#     #testing for streamlit app
#     generated_mcqs.append({
#             "question": response, # The whole LLM response for now
#             "option": ["Option A", "Option B", "Option C", "Option D"],
#             "correct_answer": "Option B" 
#         })
    
#     return generated_mcqs

def generate_mcq(topic: str, retriever, num_questions: int=5): 
    docs = retriever.invoke(topic)
    mcq_chain = mcq_prompt | llm | StrOutputParser()
    generated_mcqs = []
    
    for i in range(min(num_questions, len(docs))):
        context_chunk = docs[i].page_content
        response = mcq_chain.invoke({"context": context_chunk})
        
        # parsing the LLM output to match streamlit UI requirements 
        try:
            # 1. Robust Question Extraction
            # Splitting by A) or 1) to ensure we get only the question text
            question_part = re.split(r'[A-D]\)|1\)', response)[0]
            question = question_part.replace("Question:", "").strip()
            
            # 2. Robust Options Extraction
            # This regex looks for A) Text, B) Text, etc., OR 1) Text, 2) Text, etc.
            options = re.findall(r'(?:[A-D]\)|[1-4]\)) (.*)', response)
            
            # 3. Robust Correct Answer Extraction
            # Searches for 'Correct Answer:' followed by a Letter or Number, ignoring case
            correct_match = re.search(r'Correct Answer:\s*(?:Option\s*)?([A-D]|1|2|3|4)', response, re.IGNORECASE)
            
            if correct_match:
                ans_raw = correct_match.group(1).upper()
                # Map both letters and numbers to the correct list index
                letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
                idx = letter_to_index.get(ans_raw, 0)
                
                # Ensure the index exists in the found options
                if len(options) > idx:
                    correct_text = options[idx].strip()
                else:
                    correct_text = "Refer to Explanation"
            else:
                correct_text = "Refer to Explanation"

            # Create the dictionary your UI expects
            generated_mcqs.append({
                "question": question,
                # Fallback to letters if parsing fails to find 4 distinct options
                "option": [opt.strip() for opt in options] if len(options) >= 4 else ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": correct_text
            })
            
        except Exception as e:
            print(f"Error parsing MCQ: {e}")
            # Add a fallback dictionary to prevent the UI loop from crashing
            generated_mcqs.append({
                "question": "Error parsing this specific question.",
                "option": ["N/A", "N/A", "N/A", "N/A"],
                "correct_answer": "N/A"
            })

    return generated_mcqs


if __name__ == "__main__":

    # raw_documents = load_docs(file_path)
    # print(f"Length of the raw documents: {len(raw_documents)}")

    # text_chunks = split_docs(raw_documents)
    # print(f"Length of the text chunks: {len(text_chunks)}")

    
    # embeddings=downlode_hugging_face_embeddings()
    # #test_embedding = embeddings.embed_query("Suprito")
    # #print(len(test_embedding))

    # vectorstore=create_vectorstore(text_chunks, embeddings)

    persist_directory = 'chroma_db_MCQGen'
    embeddings=downlode_hugging_face_embeddings()
    if os.path.exists(persist_directory):
        print("Existing Vectorstore found. Loading...")
        vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
    else:
        # 2. CREATE: Only runs the first time or if you delete the folder
        print("No Vectorstore found. Starting ingestion...")
        raw_documents = load_docs(file_path)
        text_chunks = split_docs(raw_documents)
        vectorstore = create_vectorstore(text_chunks, embeddings)
        print("Vectorstore created successfully.")

    # retriver=vectorstore.as_retriever(search_kwargs={"k": 10})
    # #search_kwargs={"k": 3}
    # docs=retriver.invoke("what is Homeopathy")
    # print(docs[2].page_content)

    #llm=load_llama()

    retriver=vectorstore.as_retriever(search_kwargs={"k": 10})

    while True:
        topic = input("Enter the topic for MCQ generation (or 'q' to quit): ").strip()
        if topic.lower() == 'q':
            break
        mcqs = generate_mcq(topic, retriver, num_questions=5)
        for i, mcq in enumerate(mcqs, 1):
            print(f"\nMCQ {i}:\n{mcq}\n")


    



