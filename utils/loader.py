from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_documents(file_path):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    else:
        raise ValueError("Tipo de arquivo não suportado.")
