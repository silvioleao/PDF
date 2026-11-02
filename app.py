import threading

import numpy as np
import streamlit as st
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

st.set_page_config(page_title="Pesquisa em PDFs (sem chave)", page_icon="📄", layout="wide")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # open source, ~120 MB
LLM_OPTIONS = {
    "Qwen2.5 0.5B Instruct (mais leve)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5 1.5B Instruct (melhor, precisa de mais memória)": "Qwen/Qwen2.5-1.5B-Instruct",
}
MODE_SEARCH = "Só busca (leve, sem LLM)"
MODE_ANSWER = "Busca + resposta (modelo local pequeno)"


# ---------- Utilidades ----------
@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Carregando modelo de linguagem (1ª vez baixa os pesos)...")
def load_llm(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.eval()
    return tok, model


def extract_pages(file):
    reader = PdfReader(file)
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield i, text


def chunk_text(text, size=800, overlap=150):
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


def format_hits(hits):
    parts = []
    for i, (c, score) in enumerate(hits, start=1):
        snippet = c["text"].replace("\n", " ")
        parts.append(
            f"**{i}. {c['file']} · p.{c['page']}** (similaridade {score:.2f})\n\n> {snippet}"
        )
    return "\n\n".join(parts)


def stream_answer(model_id, question, hits, temperature, max_new_tokens):
    tok, model = load_llm(model_id)
    context = "\n\n".join(f"[{c['file']}, p.{c['page']}]\n{c['text']}" for c, _ in hits)
    context = context[:6000]  # modelos pequenos têm contexto efetivo curto
    messages = [
        {
            "role": "system",
            "content": (
                "Você responde perguntas usando SOMENTE os trechos fornecidos. "
                "Cite a fonte no formato [arquivo, p.X]. Se a resposta não estiver "
                "nos trechos, diga que não encontrou. Responda no idioma da pergunta."
            ),
        },
        {"role": "user", "content": f"Trechos:\n{context}\n\nPergunta: {question}"},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    def _run():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    threading.Thread(target=_run, daemon=True).start()
    for piece in streamer:
        yield piece


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configurações")
    mode = st.radio("Modo", [MODE_SEARCH, MODE_ANSWER])
    if mode == MODE_ANSWER:
        llm_label = st.selectbox("Modelo de linguagem", list(LLM_OPTIONS))
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.1, step=0.1)
        max_new_tokens = st.slider("Máx. tokens da resposta", 100, 800, 400, step=50)
        st.caption(
            "Em CPU a resposta é lenta, e o modelo consome bastante memória. "
            "No Streamlit Cloud gratuito, prefira o modo só busca."
        )
    st.divider()
    k = st.slider("Trechos recuperados (top-k)", 2, 10, 4)
    chunk_size = st.slider("Tamanho do chunk (caracteres)", 400, 2000, 800, step=100)
    overlap = st.slider("Sobreposição", 0, 400, 150, step=50)

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
st.title("📄 Pesquisa em PDFs (sem chave de API)")

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

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = retrieve(question, st.session_state.chunks, st.session_state.emb, k)

    with st.chat_message("assistant"):
        if mode == MODE_SEARCH:
            reply = "**Trechos mais relevantes:**\n\n" + format_hits(hits)
            st.markdown(reply)
        else:
            try:
                reply = st.write_stream(
                    stream_answer(
                        LLM_OPTIONS[llm_label], question, hits, temperature, max_new_tokens
                    )
                )
            except Exception as e:
                st.error(f"Erro ao carregar ou executar o modelo: {e}")
                st.stop()
            with st.expander("Trechos usados"):
                st.markdown(format_hits(hits))

    st.session_state.messages.append({"role": "assistant", "content": reply})
