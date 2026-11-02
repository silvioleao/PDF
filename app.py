import os

import numpy as np
import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Pesquisa em PDFs com RAG (cloud)", page_icon="📄", layout="wide")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Provedores com API de chat no formato "chat completions" (protocolo aberto
# usado por vLLM, Groq, Together etc.). O cliente é o huggingface_hub (Apache 2.0).
# Confira o catálogo de cada um:
# os nomes dos modelos mudam com frequência.
PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "Together AI": {
        "base_url": "https://api.together.xyz/v1",
        "env": "TOGETHER_API_KEY",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "Fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env": "FIREWORKS_API_KEY",
        "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
    "Hugging Face": {
        "base_url": "https://router.huggingface.co/v1",
        "env": "HF_TOKEN",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "Outro (vLLM próprio, etc.)": {
        "base_url": "http://SEU_SERVIDOR:8000/v1",
        "env": "OPENAI_API_KEY",
        "model": "",
    },
}


# ---------- Utilidades ----------
@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def extract_pages(file):
    reader = PdfReader(file)
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield i, text


def chunk_text(text, size=1000, overlap=200):
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def build_index(files, chunk_size, overlap):
    embedder = load_embedder()
    chunks = []
    for f in files:
        for page_num, text in extract_pages(f):
            for c in chunk_text(text, chunk_size, overlap):
                chunks.append({"file": f.name, "page": page_num, "text": c})
    if not chunks:
        return None, None
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return chunks, np.asarray(embeddings)


def retrieve(query, chunks, embeddings, k):
    q = load_embedder().encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


def stream_answer(client, model, question, hits, temperature):
    context = "\n\n".join(
        f"[{c['file']}, p.{c['page']}]\n{c['text']}" for c, _ in hits
    )
    system = (
        "Você responde perguntas usando SOMENTE os trechos de documentos fornecidos. "
        "Cite a fonte no formato [arquivo, p.X] após cada afirmação. "
        "Se os trechos não contiverem a resposta, diga claramente que não encontrou "
        "nas fontes. Responda no idioma da pergunta."
    )
    user = f"Trechos:\n{context}\n\nPergunta: {question}"
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=1500,
        stream=True,
    )
    for part in stream:
        if part.choices and part.choices[0].delta.content:
            yield part.choices[0].delta.content


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configurações")
    provider_name = st.selectbox("Provedor", list(PROVIDERS))
    p = PROVIDERS[provider_name]

    base_url = st.text_input("Base URL", value=p["base_url"])
    model = st.text_input("Modelo", value=p["model"])
    default_key = st.secrets.get(p["env"], os.getenv(p["env"], "")) if hasattr(st, "secrets") else ""
    api_key = st.text_input(f"API key ({p['env']})", value=default_key, type="password")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.1, step=0.1)

    st.divider()
    k = st.slider("Trechos recuperados (top-k)", 2, 15, 5)
    chunk_size = st.slider("Tamanho do chunk (caracteres)", 400, 2000, 1000, step=100)
    overlap = st.slider("Sobreposição", 0, 500, 200, step=50)

    files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Indexar PDFs", type="primary", disabled=not files):
        with st.spinner("Extraindo texto e gerando embeddings..."):
            chunks, emb = build_index(files, chunk_size, overlap)
        if chunks is None:
            st.error("Nenhum texto extraído. PDFs escaneados precisam de OCR.")
        else:
            st.session_state.update(chunks=chunks, emb=emb, messages=[])
            st.success(f"{len(chunks)} trechos indexados.")

# ---------- Chat ----------
st.title("📄 Pesquisa em PDFs com RAG")
st.caption(
    "Embeddings rodam localmente; apenas os trechos recuperados e a pergunta "
    "são enviados ao provedor do LLM."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Pergunte algo sobre seus documentos")

if question:
    if "chunks" not in st.session_state:
        st.warning("Envie e indexe os PDFs primeiro (barra lateral).")
        st.stop()
    if not api_key or not model:
        st.warning("Informe a API key e o modelo na barra lateral.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = retrieve(question, st.session_state.chunks, st.session_state.emb, k)

    with st.chat_message("assistant"):
        client = InferenceClient(base_url=base_url, api_key=api_key)
        try:
            answer = st.write_stream(stream_answer(client, model, question, hits, temperature))
        except Exception as e:
            st.error(f"Erro ao chamar o provedor: {e}")
            st.stop()
        with st.expander("Trechos usados"):
            for c, score in hits:
                st.caption(f"{c['file']} · p.{c['page']} · similaridade {score:.2f}")
                st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
