# -*- coding: utf-8 -*-


"""
Full Agentic Smart Ingest Script for EDUBOT with Watcher Agent
Maintains embeddings with agent-driven workflow using LangGraph
Auto-handles new/deleted files, validation, retries, and continuous monitoring
Updated: 2025-10-26
@author: Arun Teja
"""
import os
import pickle
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain loaders
from langchain_community.document_loaders import (
    PyPDFLoader, PyMuPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# LangGraph
from langgraph.graph import StateGraph, START, END

# Patch for unstructured bug
import pandas as pd
import pandas.api.types as ptypes
pd.numeric = ptypes

# =====================
# Image OCR imports
# =====================
try:
    from PIL import Image
    import pytesseract
    TESSERACT_INSTALLED = False
    default_tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_tess_path):
        pytesseract.pytesseract.tesseract_cmd = default_tess_path
        TESSERACT_INSTALLED = True
        print(f"✅ Tesseract found: {default_tess_path}")
    else:
        print("⚠️ Tesseract not found. Images will be skipped.")
except ImportError:
    TESSERACT_INSTALLED = False
    print("⚠️ PIL or pytesseract not installed. Images will be skipped.")

# =====================
# Configs
# =====================
DATA_PATH = r"D:\AAIDC\Project 2\Data"
DB_FAISS_PATH = r"D:\AAIDC\Project 2\vectorstore"
FILE_MAPPING_PATH = r"D:\AAIDC\Project 2\file_mapping.pkl"
LOG_FILE = r"D:\AAIDC\Project 2\ingest_log.txt"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# =====================
# Load file mapping
# =====================
file_mapping = {}
if os.path.exists(FILE_MAPPING_PATH):
    with open(FILE_MAPPING_PATH, "rb") as f:
        file_mapping = pickle.load(f)

# =====================
# Helper: Log updates
# =====================
def log_update(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

# =====================
# Universal file loader
# =====================
def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        docs = []

        if ext == ".pdf":
            loaded = False
            for loader_cls in [PyPDFLoader, PyMuPDFLoader]:
                try:
                    loader = loader_cls(file_path)
                    docs = loader.load()
                    if all(len(d.page_content.strip()) < 20 for d in docs):
                        continue
                    log_update(f"✅ Loaded PDF with {loader_cls.__name__}: {os.path.basename(file_path)}")
                    loaded = True
                    break
                except:
                    continue
            if not loaded:
                loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
                docs = loader.load()
                log_update(f"✅ Loaded PDF with OCR: {os.path.basename(file_path)}")

        elif ext in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(file_path)
            docs = loader.load()
            log_update(f"✅ Loaded PowerPoint: {os.path.basename(file_path)}")

        elif ext in [".doc", ".docx"]:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            log_update(f"✅ Loaded Word Document: {os.path.basename(file_path)}")

        elif ext in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()
            log_update(f"✅ Loaded Excel File: {os.path.basename(file_path)}")

        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            log_update(f"✅ Loaded Text File: {os.path.basename(file_path)}")

        elif ext == ".csv":
            try:
                df = pd.read_csv(file_path)
                text = df.to_csv(index=False)
                if text.strip():
                    docs = [Document(page_content=text, metadata={"source": file_path})]
                    log_update(f"✅ Loaded CSV File: {os.path.basename(file_path)}")
            except Exception as e:
                log_update(f"❌ Failed to load CSV {os.path.basename(file_path)}: {e}")

        elif ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = json.dumps(data, indent=2)
                if text.strip():
                    docs = [Document(page_content=text, metadata={"source": file_path})]
                    log_update(f"✅ Loaded JSON File: {os.path.basename(file_path)}")
            except Exception as e:
                log_update(f"❌ Failed to load JSON {os.path.basename(file_path)}: {e}")

        elif ext in [".jpg", ".jpeg", ".png"]:
            if TESSERACT_INSTALLED:
                try:
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img, lang="eng")
                    if text.strip():
                        docs = [Document(page_content=text, metadata={"source": file_path})]
                        log_update(f"✅ Loaded Image via Tesseract OCR: {os.path.basename(file_path)}")
                    else:
                        log_update(f"⚠️ Skipping {os.path.basename(file_path)} (no text detected)")
                        return []
                except Exception as e:
                    log_update(f"⚠️ Skipping image {os.path.basename(file_path)} (OCR failed: {e})")
                    return []
            else:
                log_update(f"⚠️ Skipping image {os.path.basename(file_path)} (Tesseract not installed)")
                return []

        else:
            log_update(f"⚠️ Unsupported file type: {os.path.basename(file_path)}")
            return []

        return docs

    except Exception as e:
        log_update(f"❌ Failed to load {os.path.basename(file_path)}: {e}")
        return []

# =====================
# Split documents into chunks
# =====================
def split_documents(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    clean_docs = [d for d in doc_list if len(d.page_content.strip()) > 0]
    return text_splitter.split_documents(clean_docs)

# =====================
# Main vector DB update
# =====================
def update_vector_db():
    # Supported file types
    supported_exts = [".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx", 
                      ".txt", ".jpg", ".jpeg", ".png", ".csv", ".json"]

    # Ensure FAISS folder exists
    os.makedirs(DB_FAISS_PATH, exist_ok=True)

    # List current files in DATA_PATH
    current_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)
                     if os.path.splitext(f)[1].lower() in supported_exts]
    current_files_set = set(current_files)

    # Load existing FAISS DB if index exists
    index_path = os.path.join(DB_FAISS_PATH, "index.faiss")
    if os.path.exists(index_path):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        log_update("✅ Loaded existing FAISS vector store.")
    else:
        db = None
        log_update("ℹ️ No existing FAISS vector store found. A new one will be created.")

    old_chunks = sum(info["chunks"] for info in file_mapping.values()) if file_mapping else 0

    # =======================
    # Remove deleted files
    # =======================
    removed_files = [fp for fp in list(file_mapping.keys()) if fp not in current_files_set]
    deleted_chunks = 0
    if removed_files and db:
        for fp in removed_files:
            try:
                existing_ids = set(db.index_to_docstore_id.values())
                ids_to_delete = set(file_mapping[fp]["vector_ids"])
                valid_ids = ids_to_delete & existing_ids

                if valid_ids:
                    db.delete(ids=list(valid_ids))
                    deleted_chunks += file_mapping[fp]["chunks"]
                    log_update(f"🗑️ Removed file: {os.path.basename(fp)} ({file_mapping[fp]['chunks']} chunks)")
                else:
                    log_update(f"⚠️ Skipped deletion for {os.path.basename(fp)} (IDs not found in FAISS)")
            except Exception as e:
                log_update(f"❌ Error deleting vectors for {os.path.basename(fp)}: {e}")

            del file_mapping[fp]

    # =======================
    # Add new files
    # =======================
    new_files = [fp for fp in current_files if fp not in file_mapping]
    added_chunks = 0
    if new_files:
        log_update(f"📄 New files to process: {len(new_files)}")
        documents = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {executor.submit(load_file, f): f for f in new_files}
            for future in as_completed(future_to_file):
                docs = future.result()
                if docs:
                    for d in docs:
                        d.metadata["source"] = future_to_file[future]
                    documents.extend(docs)

        if documents:
            chunks = []
            for doc in documents:
                chunks.extend(split_documents([doc]))
            added_chunks = len(chunks)

            if added_chunks > 0:
                if db is None:
                    db = FAISS.from_documents(chunks, embeddings)
                    ids = list(range(len(chunks)))
                    log_update(f"✅ Created new FAISS vector store with {added_chunks} chunks.")
                else:
                    ids = db.add_documents(chunks)
                    log_update(f"✅ Added {added_chunks} chunks to existing FAISS vector store.")

                # Map chunks to files
                idx = 0
                for fp in new_files:
                    related_chunks = [c for c in chunks if c.metadata.get("source") == fp]
                    count = len(related_chunks)
                    assigned_ids = ids[idx:idx + count]
                    idx += count
                    file_mapping[fp] = {"vector_ids": assigned_ids, "chunks": count}
                    log_update(f"✅ Added file: {os.path.basename(fp)} ({count} chunks)")
        else:
            log_update("⚠️ No new documents loaded.")
    else:
        log_update("ℹ️ No new files to process.")

    # =======================
    # Save updated DB and mapping
    # =======================
    if db:
        db.save_local(DB_FAISS_PATH)
    with open(FILE_MAPPING_PATH, "wb") as f:
        pickle.dump(file_mapping, f)

    final_chunks = sum(info["chunks"] for info in file_mapping.values()) if file_mapping else 0

    # Summary
    log_update("===== SUMMARY =====")
    log_update(f"Old Chunks: {old_chunks}")
    log_update(f"Added Chunks: {added_chunks}")
    log_update(f"Deleted Chunks: {deleted_chunks}")
    log_update(f"Final Total Chunks: {final_chunks}")
    log_update("====================")

    return {"old": old_chunks, "added": added_chunks, "deleted": deleted_chunks, "final": final_chunks}

# =====================
# Agentic Workflow with LangGraph
# =====================
def detect_files(state):
    supported_exts = [".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx", 
                      ".txt", ".jpg", ".jpeg", ".png", ".csv", ".json"]
    files = [f for f in os.listdir(DATA_PATH) if os.path.splitext(f)[1].lower() in supported_exts]
    log_update(f"🔍 Detected {len(files)} supported files.")
    return {"files": files}

def ingest_files(state):
    result = update_vector_db()
    return {"ingest_result": result}

def validate_ingest(state):
    if os.path.exists(DB_FAISS_PATH):
        log_update("✅ Validation passed: FAISS DB exists.")
        return {"status": "valid"}
    else:
        log_update("❌ Validation failed: FAISS DB missing!")
        return {"status": "error"}

# Build LangGraph workflow with proper START edge
graph = StateGraph(dict)
graph.add_node("detect", detect_files)
graph.add_node("ingest", ingest_files)
graph.add_node("validate", validate_ingest)

# Correct edges including START → first node
graph.add_edge(START, "detect")
graph.add_edge("detect", "ingest")
graph.add_edge("ingest", "validate")
graph.add_edge("validate", END)

app = graph.compile()

# =====================
# Watcher Agent
# =====================
def watcher_agent(poll_interval=15):
    previous_files = set(os.listdir(DATA_PATH))
    log_update("👀 Watcher Agent started...")

    # First-run ingestion: only for files not already in file_mapping
    new_files = [f for f in previous_files if os.path.join(DATA_PATH, f) not in file_mapping]
    if new_files:
        log_update(f"📄 First-run: processing {len(new_files)} new files...")
        result = app.invoke({})
        log_update(f"🤖 Initial ingest triggered: {result}")
    else:
        log_update("ℹ️ First-run: all existing files already ingested, skipping.")

    try:
        while True:
            current_files = set(os.listdir(DATA_PATH))
            added = current_files - previous_files
            removed = previous_files - current_files

            if added or removed:
                if added:
                    log_update(f"➕ Detected new files: {list(added)}")
                if removed:
                    log_update(f"🗑️ Detected deleted files: {list(removed)}")

                result = app.invoke({})
                log_update(f"🤖 Agentic ingest triggered by Watcher: {result}")

            previous_files = current_files
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        log_update("🛑 Watcher Agent stopped by user.")

# =====================
# Run Agentic Ingest or Watcher
# =====================
if __name__ == "__main__":
    watcher_agent(poll_interval=5)


