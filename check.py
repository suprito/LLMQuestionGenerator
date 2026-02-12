import torch
from langchain_community.document_loaders import PyPDFLoader

print(f"✅ Torch version: {torch.__version__}")
print(f"✅ Torch loaded from: {torch.__file__}")
print("✅ LangChain imports successful!")

# Quick check if it can see your PDF
import os
pdf_path = "data/MedicalBook.pdf"
print(f"📂 PDF exists: {os.path.exists(pdf_path)}")