import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import pdfplumber
from docx import Document as DocxDocument
from tqdm import tqdm

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("Clearing database")
        clear_database()

    documents = []
    documents += load_pdf_documents()
    documents += load_word_documents()

    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_pdf_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                documents.append(Document(page_content=text, metadata={"source": file}))
    return documents


def load_word_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".docx") or file.endswith(".doc"):
            doc_path = os.path.join(DATA_PATH, file)
            try:
                doc = DocxDocument(doc_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                documents.append(Document(page_content=text, metadata={"source": file}))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Existing document count in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new documents")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        for chunk, chunk_id in tqdm(zip(new_chunks, new_chunk_ids), total=len(new_chunks), desc="Adding chunks"):
            db.add_documents([chunk], ids=[chunk_id])

        print("Finished updating the database")
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
