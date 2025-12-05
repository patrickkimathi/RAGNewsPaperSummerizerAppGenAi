# app.py
# ============================================================
# MyGov Intelligence Assistant — Full Streamlit App
# ============================================================
#
# Requirements (install in your environment, NOT in this file):
# pip install streamlit langchain langchain-community sentence-transformers faiss-cpu transformers torch PyPDF2 beautifulsoup4 scikit-learn pandas matplotlib spacy fpdf huggingface-hub
# python -m spacy download en_core_web_sm
#
# Then run:
# streamlit run app.py
# ============================================================

import streamlit as st
import os
from pathlib import Path
from io import BytesIO
import time
from collections import Counter
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

# Clustering and NER
from sklearn.cluster import KMeans
import spacy

# PDF export
from fpdf import FPDF

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="MyGov RAG + Dashboard", layout="wide")
st.title("🇰🇪 MyGov Intelligence Assistant — RAG, Analytics & Summaries")

# Create folders for uploads and scraped PDFs
os.makedirs("uploaded_mygov", exist_ok=True)
os.makedirs("scraped_mygov", exist_ok=True)

# ---------------- Load spaCy NER ----------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        raise

nlp = load_spacy()

# ---------------- Load LLM ----------------
@st.cache_resource
def load_llm():
    """
    Loads a seq2seq model (FLAN-T5 or similar) and wraps it in a LangChain HuggingFacePipeline.
    NOTE: large models may be slow or require GPU.
    """
    model_name = "google/flan-t5-large"  # change to flan-t5-base if resource-constrained
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to load LLM model {model_name}: {e}")
        raise

with st.spinner("Loading language model (may take a while)..."):
    llm = load_llm()

# ---------------- Load Embeddings ----------------
@st.cache_resource
def load_embeddings():
    """
    Sentence-transformers embedding model used for vectorstore.
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        st.error(f"Failed to load embeddings model {model_name}: {e}")
        raise

with st.spinner("Loading embeddings..."):
    embeddings = load_embeddings()

# ---------------- Sidebar: Upload or Scrape PDFs ----------------
st.sidebar.header("1) Upload or Scrape MyGov Issues")
uploaded = st.sidebar.file_uploader("Upload MyGov PDFs (multiple allowed)", type="pdf", accept_multiple_files=True)

if uploaded:
    saved = 0
    for f in uploaded:
        safe_name = f.name.replace("/", "_")
        path = Path("uploaded_mygov") / safe_name
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved += 1
    st.sidebar.success(f"{saved} file(s) saved to uploaded_mygov/")

st.sidebar.subheader("Auto-scraper (optional)")
scrape_url = st.sidebar.text_input("Page URL that lists MyGov PDFs (example page with links)")
scrape_button = st.sidebar.button("Run Auto-Scraper (save PDFs)")

if scrape_button and scrape_url:
    st.sidebar.info("Running scraper — please ensure the page allows scraping.")
    try:
        resp = requests.get(scrape_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        links = [a.get("href") for a in soup.find_all("a", href=True) if a.get("href").lower().endswith(".pdf")]
        st.sidebar.write(f"Found {len(links)} PDF links (downloaded to scraped_mygov/).")
        for link in links:
            if link.startswith("/"):
                base = "{0.scheme}://{0.netloc}".format(requests.utils.urlparse(scrape_url))
                link = base + link
            r = requests.get(link, timeout=30)
            r.raise_for_status()
            filename = link.split("/")[-1]
            filename = filename.split("?")[0]
            with open(os.path.join("scraped_mygov", filename), "wb") as out:
                out.write(r.content)
        st.sidebar.success("Scrape complete.")
    except Exception as e:
        st.sidebar.error(f"Scrape failed: {e}")

# ---------------- Build / Refresh Index ----------------
st.header("2) Build / Refresh Index")
rebuild = st.button("Build/Refresh Index")

if rebuild:
    st.info("Loading PDFs and building index... please wait.")
    all_pages = []
    for folder in ("uploaded_mygov", "scraped_mygov"):
        for file in os.listdir(folder):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(folder, file)
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    for p in pages:
                        p.metadata["source_file"] = file
                    all_pages.extend(pages)
                except Exception as e:
                    st.warning(f"Failed to load {file}: {e}")
    if len(all_pages) == 0:
        st.warning("No PDFs found. Upload or scrape first.")
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(all_pages)
        st.write(f"Chunks created: {len(chunks)}")
        with st.spinner("Creating vectorstore (FAISS)..."):
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local("faiss_mygov_index")
        st.success("Index built and saved.")
        st.session_state["chunks"] = chunks
        st.session_state["vectorstore"] = vectorstore

# If vectorstore exists on disk, load it
if "vectorstore" not in st.session_state:
    if Path("faiss_mygov_index").exists():
        try:
            vectorstore = FAISS.load_local("faiss_mygov_index", embeddings)
            st.session_state["vectorstore"] = vectorstore
            st.info("Loaded saved vectorstore from disk.")
        except Exception as e:
            st.warning(f"Failed to load saved vectorstore: {e}")

# ---------------- Topic-filtered Search & RAG Q&A ----------------
st.header("3) Topic-filtered Search & RAG Q&A")
topic_options = ["All", "Agriculture", "Health", "Education", "Infrastructure", "Jobs", "Tenders"]
topic = st.selectbox("Select topic filter", topic_options)
query = st.text_input("Ask a question about the newspapers:")
search_btn = st.button("Search (RAG)")

def safe_llm_call(prompt_text):
    """
    Call the LLM and normalize output to a string.
    The LLM wrapper sometimes returns a dict, list, or string depending on implementation.
    """
    try:
        out = llm(prompt_text)
    except Exception as e:
        st.error(f"LLM generation failed: {e}")
        return ""
    # normalize common return formats
    if isinstance(out, list) and len(out) > 0:
        first = out[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        elif isinstance(first, str):
            return first
        else:
            return str(first)
    elif isinstance(out, dict) and "generated_text" in out:
        return out["generated_text"]
    elif isinstance(out, str):
        return out
    else:
        return str(out)

if search_btn:
    if "vectorstore" not in st.session_state:
        st.error("Build the index first.")
    elif query.strip() == "":
        st.warning("Type a question first.")
    else:
        biased_query = f"{topic}. {query}" if topic != "All" else query
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6})
        docs = retriever.get_relevant_documents(biased_query)
        if not docs:
            st.warning("No relevant documents found.")
        else:
            st.subheader("Retrieved Snippets")
            for i, d in enumerate(docs):
                st.markdown(f"**Snippet {i+1}** — _{d.metadata.get('source_file','unknown')}_")
                st.write(d.page_content[:400] + "...")
                st.write("---")
            context = "\n".join([d.page_content for d in docs])
            prompt = f"""You are a factual assistant. Use ONLY context to answer.

Context:
{context}

Question: {query}
Answer:"""
            with st.spinner("Generating answer..."):
                text = safe_llm_call(prompt)
            st.subheader("Answer (RAG)")
            st.write(text)
            st.session_state["last_retrieved_docs"] = docs

# ---------------- Auto-clustering ----------------
st.header("4) Cluster Retrieved Snippets")
cluster_btn = st.button("Cluster last retrieved snippets")
if cluster_btn:
    if "last_retrieved_docs" not in st.session_state:
        st.warning("Run a search first.")
    else:
        docs = st.session_state["last_retrieved_docs"]
        texts = [d.page_content for d in docs]
        # create vectors for clustering
        try:
            vectors = embeddings.embed_documents(texts)
        except Exception as e:
            st.error(f"Embedding documents for clustering failed: {e}")
            vectors = []
        if len(vectors) == 0:
            st.warning("No vectors produced for clustering.")
        else:
            X = np.array(vectors)
            k = min(4, max(1, len(texts)))
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)
            st.subheader("Clusters:")
            for cid in range(k):
                st.markdown(f"### Cluster {cid+1}")
                cluster_texts = [texts[i] for i, lab in enumerate(labels) if lab == cid]
                for t in cluster_texts:
                    st.write(t[:400] + "...")
                    st.write("---")
            st.session_state["clusters"] = {"labels": labels.tolist(), "texts": texts}

# ---------------- Named Entity Recognition ----------------
st.header("5) Named Entity Recognition")
ner_btn = st.button("Extract Named Entities")

if ner_btn:
    if "last_retrieved_docs" in st.session_state:
        source_texts = [d.page_content for d in st.session_state["last_retrieved_docs"]]
    elif "chunks" in st.session_state:
        source_texts = [c.page_content for c in st.session_state["chunks"]]
    else:
        st.warning("No text available.")
        source_texts = []
    full = " ".join(source_texts)[:200000]  # cap to avoid huge NER runs
    if full.strip() == "":
        st.warning("No text to run NER on.")
    else:
        doc = nlp(full)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        gpe = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        st.subheader("Top People")
        st.write(pd.DataFrame(Counter(persons).most_common(20), columns=["Person", "Count"]))
        st.subheader("Top Organizations")
        st.write(pd.DataFrame(Counter(orgs).most_common(20), columns=["Org", "Count"]))
        st.subheader("Top Locations / Counties")
        st.write(pd.DataFrame(Counter(gpe).most_common(20), columns=["Location", "Count"]))
        st.session_state["ner"] = {"persons": Counter(persons).most_common(20),
                                   "orgs": Counter(orgs).most_common(20),
                                   "gpe": Counter(gpe).most_common(20)}

# ---------------- Full Summary + PDF Export ----------------
st.header("6) Full Newspaper Summary & PDF Export")
summary_btn = st.button("Generate Full Summary + PDF")

if summary_btn:
    if "chunks" in st.session_state:
        texts = [c.page_content for c in st.session_state["chunks"]]
    elif "last_retrieved_docs" in st.session_state:
        texts = [d.page_content for d in st.session_state["last_retrieved_docs"]]
    else:
        st.error("No text available.")
        texts = []
    if len(texts) == 0:
        st.warning("No text to summarize.")
    else:
        # We limit the amount of text sent to the LLM to avoid excessive prompt size
        max_chars = 30000
        joined = "\n".join(texts)
        full_text = joined[:max_chars]
        summ_prompt = f"""You are a Kenyan news summarizer. Produce a structured summary with headlines and bullets, covering national announcements, projects, education, health, agriculture, tenders/jobs.

Text:
{full_text}"""
        with st.spinner("Summarizing..."):
            summary_text = safe_llm_call(summ_prompt)
        st.subheader("Summary")
        st.write(summary_text)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "MyGov Issues — AI Summary", ln=True, align="C")
        pdf.set_font("Arial", size=11)
        for line in summary_text.split("\n"):
            pdf.multi_cell(0, 6, line)
        ner_info = st.session_state.get("ner", {})
        if ner_info:
            pdf.ln(4)
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, "Top People:", ln=True)
            pdf.set_font("Arial", size=10)
            for p, c in ner_info.get("persons", [])[:10]:
                pdf.cell(0, 6, f"{p} — {c}", ln=True)
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("⬇ Download Summary PDF", pdf_bytes, file_name="MyGov_Summary.pdf", mime="application/pdf")

# ---------------- Quick Counts Dashboard ----------------
st.header("7) Quick Dashboard")
dashboard_btn = st.button("Generate Quick Counts")
if dashboard_btn:
    if "chunks" in st.session_state:
        full_text = " ".join([c.page_content for c in st.session_state["chunks"]]).lower()
    else:
        full_text = ""
    job_keywords = ["job", "vacancy", "recruit", "career", "position", "apply"]
    tender_keywords = ["tender", "procurement", "bid", "proposal", "supply"]
    st.metric("Job mentions", sum(full_text.count(k) for k in job_keywords))
    st.metric("Tender mentions", sum(full_text.count(k) for k in tender_keywords))
    if "ner" in st.session_state:
        gpe = st.session_state["ner"]["gpe"]
        if gpe:
            names, counts = zip(*gpe[:10])
            df = pd.DataFrame({"location": names, "count": counts})
            st.bar_chart(df.set_index("location"))

# ---------------- Helpful status info ----------------
st.sidebar.markdown("---")
st.sidebar.write("Index files (faiss_mygov_index/) and uploaded PDFs (uploaded_mygov/) are stored in the app folder.")
st.sidebar.write("If deploying to Streamlit Cloud, include a requirements.txt with the packages listed at the top of this file.")
