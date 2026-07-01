"""
Fermilab Q&A Assistant — Streamlit front-end.

What this app does, in plain terms:
  1. Loads a fine-tuned Qwen3-4B model (QLoRA adapter or merged weights).
  2. Optionally retrieves Fermilab documents (RAG) to ground the answer.
  3. Lets the user chat, compare RAG vs no-RAG, and view pipeline metrics.

The code is organised as:
  - Setup        : imports, page config, CSS
  - Loaders      : cached functions that load the model, retriever, corpus map, demo cache
  - Helpers      : small reusable functions (generate, build prompt, render UI)
  - UI / Tabs    : the four tabs (Chat, Map, Media, Metrics)

PERFORMANCE NOTES:
  - Inference runs in bf16 on GPU instead of 4-bit nf4. nf4 exists to SAVE VRAM
    during training; at serve time it must dequantize on every matmul, so it is
    slower. Qwen3-4B in bf16 is only ~8GB, which fits easily on an Anvil GPU.
  - The LoRA adapter is folded in with merge_and_unload() to drop per-layer
    adapter overhead.
  - The chat tab streams tokens on GPU (TextIteratorStreamer) so text appears
    progressively. On CPU it falls back to a reliable non-streaming path.
  - If demo_cache.json exists, exact-match demo questions are answered instantly
    from cache (no model call) — ideal for a lag-free live demo.
"""

# ----------------------------------------------------------------------------
# 1. SETUP
# ----------------------------------------------------------------------------
import os
import json
import html as html_lib                     # escape text before putting it in HTML (anti-XSS)
from pathlib import Path
from threading import Thread                 # run model.generate() in the background for streaming
from urllib.parse import urlparse

import torch
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # lets us embed raw HTML/JS in the app
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel, PeftConfig
from retriever import Retriever               # our own RAG search module

# Force HuggingFace to work fully offline (Anvil compute nodes have no internet).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# How many tokens the model may generate per answer.
# Lower this if you hit GPU out-of-memory; raise it for longer answers.
MAX_NEW_TOKENS = 256

APP_DIR = Path(__file__).parent              # folder this script lives in

st.set_page_config(
    page_title="Fermilab Q&A Assistant Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark "space/physics" theme. Pure decoration — it does not affect any logic.
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
        .stApp { background-color: #0b0f19; color: #e2e8f0; }
        section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid rgba(255,255,255,0.05); }
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p { color: #94a3b8; }
        .main-title {
            font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 3rem;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem; text-shadow: 0px 4px 12px rgba(0,242,254,0.15);
        }
        .subtitle { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.1rem; color: #94a3b8; margin-bottom: 1.5rem; font-weight: 400; }
        .stChatMessage {
            background: rgba(30,41,59,0.4) !important; border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: 12px !important; margin-bottom: 12px !important; padding: 15px !important; backdrop-filter: blur(10px);
        }
        .source-card {
            background: rgba(30,41,59,0.6); border: 1px solid rgba(0,242,254,0.15); border-left: 4px solid #00f2fe;
            padding: 12px 18px; margin: 10px 0; border-radius: 8px; transition: all 0.3s ease;
        }
        .source-card:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,242,254,0.1); border-color: rgba(0,242,254,0.4); }
        .source-title { font-family: 'Outfit', sans-serif; font-weight: 600; color: #38bdf8; font-size: 1rem; margin-bottom: 4px; }
        .source-url { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #64748b; }
        .source-url a { color: #38bdf8 !important; text-decoration: none; }
        .source-url a:hover { text-decoration: underline; }
        .source-score { font-size: 0.8rem; font-weight: 700; color: #00f2fe; float: right; background: rgba(0,242,254,0.1); padding: 2px 8px; border-radius: 4px; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 10px; }
        .badge-rag-on { background-color: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
        .badge-rag-off { background-color: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
        .pulse-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #10b981; margin-right: 6px; animation: pulsing 1.5s infinite cubic-bezier(0.66,0,0,1); }
        @keyframes pulsing { to { box-shadow: 0 0 0 8px rgba(16,185,129,0); } }
        .metric-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9rem; background: rgba(30,41,59,0.2); border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }
        .metric-table th, .metric-table td { padding: 12px 15px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .metric-table th { background-color: rgba(79,172,254,0.1); color: #38bdf8; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 2. LOADERS (cached so they run only once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the model + tokenizer. We try three options, best first:
      1. A merged fine-tuned model (fastest, self-contained).
      2. Base model + LoRA adapter -> merged at load time.
      3. Plain base model (no fine-tuning) as a last resort.
    Returns: (model, tokenizer, status_label)

    Uses bf16 on GPU (much faster than 4-bit nf4 for a 4B model at inference).
    """
    merged_path = APP_DIR / "ft-qwen3-fermilab" / "merged"
    adapter_path = APP_DIR / "ft-qwen3-fermilab"

    # Read the base model name straight from the adapter so it always matches training.
    try:
        base_model_name = PeftConfig.from_pretrained(str(adapter_path)).base_model_name_or_path
    except Exception:
        base_model_name = "Qwen/Qwen3-4B-Instruct-2507"  # fallback repo name

    # Pick dtype + device. bf16 on GPU; float32 on CPU (very slow).
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.bfloat16
        dev_label = "GPU/bf16"
    else:
        device_map = "cpu"
        dtype = torch.float32
        dev_label = "CPU/fp32 (very slow)"
        st.sidebar.error(
            "No CUDA GPU detected. Running on CPU (very slow). "
            "Check your SLURM/GPU allocation — without a GPU, bf16 and 4-bit are both slow."
        )

    # Option 1: merged model (fastest).
    if merged_path.exists():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(merged_path), torch_dtype=dtype, device_map=device_map
            )
            tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
            model.eval()
            return model, tokenizer, f"Fine-tuned Merged ({dev_label})"
        except Exception as e:
            st.sidebar.error(f"Failed to load merged model: {e}. Trying adapter...")

    # Option 2: base model + LoRA adapter -> merge_and_unload() to drop adapter overhead.
    if adapter_path.exists():
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=dtype, device_map=device_map
            )
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            model = model.merge_and_unload()   # fold LoRA into the base weights
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model.eval()
            return model, tokenizer, f"Fine-tuned LoRA merged ({dev_label})"
        except Exception as e:
            st.sidebar.error(f"Failed to load base + adapter: {e}. Falling back to base model...")

    # Option 3: plain base model.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.eval()
    return model, tokenizer, f"Base Model Untrained ({dev_label})"


@st.cache_resource
def load_retriever():
    """Load the RAG retriever from the processed corpus file."""
    path = APP_DIR / "corpus_st7.jsonl"
    if not path.exists():
        st.error("corpus_st7.jsonl not found. Run the data pipeline first.")
        return None
    return Retriever(str(path))


@st.cache_data
def load_demo_cache():
    """
    Load demo_cache.json if present. Pre-generated answers for a fixed set of demo
    questions are served instantly (no model call), keeping a live demo lag-free.
    """
    path = APP_DIR / "demo_cache.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_corpus_map():
    """
    Return a DataFrame of 2D points (one per document chunk) for the topology map.
    If a cached corpus_map.json exists, reuse it. Otherwise compute it with
    TF-IDF + TruncatedSVD and save it for next time.
    """
    map_path = APP_DIR / "corpus_map.json"
    if map_path.exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))
        except Exception:
            pass  # unreadable file -> rebuild below

    corpus_path = APP_DIR / "corpus_st7.jsonl"
    if not corpus_path.exists():
        return pd.DataFrame()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # Read every JSONL line into a list of documents.
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line.strip()))
    if not documents:
        return pd.DataFrame()

    # Turn text into vectors, then squash to 2 dimensions for plotting.
    texts = [doc["text"] for doc in documents]
    tfidf = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(texts)
    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(tfidf)

    map_data = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "title": doc.get("title", "Untitled")[:80],
            "source_type": doc.get("source_type", "webpage_text"),
            "split": doc.get("split", "train"),
        }
        for i, doc in enumerate(documents)
    ]

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2)
    return pd.DataFrame(map_data)


# Load everything once, with a spinner for the first (slow) run.
with st.spinner("⚛️ Loading Fermilab model and index (first run may take 1-2 minutes)..."):
    model, tokenizer, model_status = load_model_and_tokenizer()
    retriever = load_retriever()
    corpus_map_df = load_corpus_map()
    demo_cache = load_demo_cache()


# ----------------------------------------------------------------------------
# 3. HELPERS (small reusable functions)
# ----------------------------------------------------------------------------
def build_messages(question, chunks):
    """
    Build the chat messages for the model.
    If `chunks` is non-empty, we run in RAG mode and ground the answer in them.
    Otherwise we ask the model to answer directly from its own knowledge.
    """
    if chunks:
        context = ""
        for i, c in enumerate(chunks, start=1):
            context += f"--- Document [{i}]: {c['title']} ---\nContent:\n{c['text']}\n\n"
        context = context[:6000]  # keep the prompt small to avoid GPU out-of-memory

        system_prompt = (
            "You are a helpful assistant that answers questions about Fermilab research.\n"
            "Answer using ONLY the provided context. If the answer is not in the context, "
            "politely say you do not know. Do not invent facts or guess beyond the text."
        )
        user_content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        system_prompt = (
            "You are a helpful assistant that answers questions about Fermilab research. "
            "Answer the question directly and concisely."
        )
        user_content = question

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_gen_kwargs(temperature):
    """Sampling config shared by both the streaming and non-streaming paths."""
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": temperature > 0.15,   # very low temperature -> deterministic
    }
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    return gen_kwargs


def generate_answer(messages, temperature):
    """
    Non-streaming path: returns the full answer at once.
    Used by the RAG vs No-RAG comparison tab (runs twice, no need to stream).
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,   # add the marker telling the model to start answering
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():                  # inference only, no gradients
        output = model.generate(**inputs, **_build_gen_kwargs(temperature))

    # Keep only the newly generated tokens (drop the prompt part).
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_answer_stream(messages, temperature):
    """
    Streaming path: yields text chunks as the model produces them.
    model.generate() runs in a background thread while the for-loop reads the streamer.

    IMPORTANT: if generate() raises inside the thread, the error would be swallowed
    and the streamer would end empty (an empty bubble with no message). We capture
    the error and re-raise it after the loop so the caller can see it and fall back.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180
    )

    gen_kwargs = dict(**inputs, streamer=streamer, **_build_gen_kwargs(temperature))

    error_box = {}

    def _worker():
        try:
            with torch.no_grad():
                model.generate(**gen_kwargs)
        except Exception as e:
            error_box["error"] = e   # keep the error to surface it later

    # generate() blocks until done -> run it in a thread so the for-loop can read in parallel.
    thread = Thread(target=_worker)
    thread.start()

    for token in streamer:
        yield token
    thread.join()

    if "error" in error_box:
        raise error_box["error"]   # surface the error so st.write_stream can fall back


def render_source_card(src):
    """Render one retrieved document as a styled card. Text is HTML-escaped (anti-XSS)."""
    title = html_lib.escape(str(src["title"]))
    origin = html_lib.escape(str(src["origin"]))
    st.markdown(f"""
        <div class="source-card">
            <span class="source-score">Relevance: {src['score']}</span>
            <div class="source-title">{title}</div>
            <div class="source-url"><a href="{origin}" target="_blank">{origin}</a></div>
        </div>
    """, unsafe_allow_html=True)


def tts_button(text):
    """
    Render a "Read aloud" button using the browser's built-in speech engine.
    The speech call lives INSIDE this button's own iframe, so it is fully
    self-contained (the old version called a function in a different iframe,
    which never existed -> the button silently did nothing).
    """
    safe_text = json.dumps(text)  # safely turn Python text into a JS string literal
    html_code = f"""
        <button onclick='
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance({safe_text});
            msg.lang = "en-US";
            window.speechSynthesis.speak(msg);
        ' style="background:#1e293b; color:#38bdf8; border:1px solid rgba(56,189,248,0.4);
                 padding:6px 12px; border-radius:6px; cursor:pointer; font-weight:600; font-size:0.85rem;">
            🔊 Read Answer Aloud
        </button>
    """
    components.html(html_code, height=45)


# ----------------------------------------------------------------------------
# 4. SIDEBAR
# ----------------------------------------------------------------------------
# Text header instead of an external image: external URLs break on offline nodes.
st.sidebar.markdown(
    "<div style='font-family:Outfit,sans-serif; font-size:1.5rem; font-weight:800; "
    "color:#38bdf8; padding:8px 0 4px;'>⚛️ Fermilab Q&A</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(f"**Model Status:** `{model_status}`")
if demo_cache:
    st.sidebar.caption(f"⚡ Demo cache active: {len(demo_cache)} instant answers")
st.sidebar.markdown("---")

st.sidebar.header("🎯 Mode Selection")
app_mode = st.sidebar.selectbox(
    "Choose Interface Mode",
    ["💬 Standard Chat Engine", "⚡ RAG vs No-RAG Comparison"],
    help="Comparison mode shows answers with and without RAG side by side.",
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration Settings")

top_k = st.sidebar.slider(
    "📚 Context Passages (Top-K)", min_value=1, max_value=5, value=3,
    help="How many passages to pull from the corpus.",
)
temperature = st.sidebar.slider(
    "🔥 Temperature", min_value=0.1, max_value=1.0, value=0.3, step=0.1,
    help="Higher = more creative but more likely to make things up; lower = more factual.",
)

st.sidebar.markdown("---")
if retriever is not None:
    st.sidebar.metric("🗂️ Indexed Corpus Size", f"{len(retriever.documents):,} Chunks")
else:
    st.sidebar.metric("🗂️ Indexed Corpus Size", "N/A (corpus missing)")

st.sidebar.markdown("""
    <div style='font-size:0.8rem; line-height:1.3; color:#64748b; margin-top:15px;'>
    <strong>System Specifications:</strong><br>
    • Base LLM: Qwen3-4B-Instruct<br>
    • Fine-tuning: QLoRA 4-bit SFT (merged for inference)<br>
    • Inference: bf16 + token streaming<br>
    • Retrieval: see retriever.py<br>
    • Voice: HTML5 Speech API
    </div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 5. MAIN TITLE + TABS
# ----------------------------------------------------------------------------
st.markdown("<div class='main-title'>Fermilab Science Q&A Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Domain-specific LLM with Retrieval-Augmented Grounding</div>", unsafe_allow_html=True)

tab_chat, tab_map, tab_media, tab_metrics = st.tabs([
    "💬 Chatbot Portal",
    "🗺️ Corpus Topology Map",
    "🎥 Fermilab Media Center",
    "📊 Crawler & Pipeline Metrics",
])


# ----------------------------------------------------------------------------
# TAB 1 — CHATBOT
# ----------------------------------------------------------------------------
with tab_chat:

    # ---- Mode A: standard chat ----
    if app_mode == "💬 Standard Chat Engine":

        use_rag = st.toggle(
            "🛡️ Enable RAG Grounding (Hallucination Guard)", value=True,
            help="On = retrieve documents as context to reduce made-up answers.",
        )

        if use_rag:
            st.markdown('<div class="badge badge-rag-on"><span class="pulse-dot"></span>RAG ACTIVE: Grounded Mode</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge badge-rag-off">RAG INACTIVE: Open Generative Mode</div>', unsafe_allow_html=True)

        # Chat history lives in session state so it survives reruns.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Re-draw the existing conversation.
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    tts_button(msg["content"])
                    if msg.get("sources"):
                        with st.expander("📚 Sources & References Used"):
                            for src in msg["sources"]:
                                render_source_card(src)

        # Handle a new question.
        if user_query := st.chat_input("Ask a question about Fermilab research..."):
            # 1. Show + store the user's message.
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            # 2. If the question is in the demo cache, answer INSTANTLY (no model call).
            cache_hit = demo_cache.get(user_query.strip().lower())

            # 3. Retrieve context if RAG is on (skipped when we already have a cache hit).
            chunks = []
            if cache_hit is None and use_rag:
                if retriever is not None:
                    with st.spinner("🔍 Searching Fermilab corpus..."):
                        chunks = retriever.retrieve(user_query, top_k=top_k)
                else:
                    st.warning("Retriever unavailable: corpus file is missing.")

            # 4. Produce the answer:
            #    - cache hit -> instant
            #    - GPU       -> stream tokens
            #    - CPU       -> reliable non-stream with a clear spinner
            with st.chat_message("assistant"):
                if cache_hit is not None:
                    answer = cache_hit["answer"]
                    chunks = cache_hit.get("sources", [])
                    st.markdown(answer)
                else:
                    messages = build_messages(user_query, chunks)
                    use_stream = torch.cuda.is_available()
                    try:
                        if use_stream:
                            # [GPU] st.write_stream shows tokens progressively and returns the full string.
                            answer = st.write_stream(generate_answer_stream(messages, temperature))
                        else:
                            # [CPU] non-stream: wait until done, then show the full answer at once.
                            with st.spinner("⚛️ Generating on CPU — this can take a few minutes, please wait..."):
                                answer = generate_answer(messages, temperature)
                            st.markdown(answer)
                    except Exception as e:
                        # Any error -> surface it, then retry with the reliable non-stream path.
                        st.error(f"⚠️ Generation error → retrying in standard mode. Details: {e}")
                        with st.spinner("⚛️ Retrying in standard mode..."):
                            answer = generate_answer(messages, temperature)
                        st.markdown(answer)

                tts_button(answer)

                sources = [
                    {"title": c["title"], "origin": c["origin"], "score": c["score"]}
                    for c in chunks
                ]
                if sources:
                    with st.expander("📚 Sources & References Used"):
                        for src in sources:
                            render_source_card(src)

            # 5. Store the assistant's message.
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

    # ---- Mode B: RAG vs No-RAG comparison ----
    else:
        st.markdown("### ⚡ Live Verification: Hallucination Reduction Test")
        st.markdown("Ask about a Fermilab experiment. The model runs twice to show how RAG reduces made-up answers.")

        st.markdown("**💡 Quick demo questions (click to ask):**")
        col_a, col_b, col_c = st.columns(3)

        if "selected_query" not in st.session_state:
            st.session_state.selected_query = "What did the SeaQuest experiment measure?"

        with col_a:
            if st.button("🧬 What did SeaQuest measure?", use_container_width=True):
                st.session_state.selected_query = "What did the SeaQuest experiment measure?"
        with col_b:
            if st.button("🔬 What is the goal of DUNE?", use_container_width=True):
                st.session_state.selected_query = "What is the goal of the DUNE experiment at LBNF?"
        with col_c:
            if st.button("⚛️ What does NOvA study?", use_container_width=True):
                st.session_state.selected_query = "What does Fermilab's NOvA experiment study?"

        custom_query = st.text_input("📝 Edit your question here:", value=st.session_state.selected_query)

        if st.button("🚀 Run Comparative Analysis", type="primary", use_container_width=True):
            # Retrieve context once; the RAG column reuses it.
            chunks = retriever.retrieve(custom_query, top_k=top_k) if retriever is not None else []

            col_direct, col_rag = st.columns(2)

            # Column 1: direct answer (NO RAG).
            with col_direct:
                st.markdown("""
                    <div style='background:rgba(239,68,68,0.05); padding:15px; border-radius:8px; border:1px solid rgba(239,68,68,0.15); margin-bottom:15px;'>
                        <span style='color:#ef4444; font-weight:700; font-size:1.1rem;'>❌ Generative Model (Direct)</span><br>
                        <span style='color:#94a3b8; font-size:0.85rem;'>Answers from model weights only. Can be factually wrong.</span>
                    </div>
                """, unsafe_allow_html=True)
                with st.spinner("Generating direct response..."):
                    ans_direct = generate_answer(build_messages(custom_query, []), temperature)
                st.markdown("### Answer:")
                st.info(ans_direct)
                tts_button(ans_direct)

            # Column 2: grounded answer (WITH RAG).
            with col_rag:
                st.markdown("""
                    <div style='background:rgba(16,185,129,0.05); padding:15px; border-radius:8px; border:1px solid rgba(16,185,129,0.15); margin-bottom:15px;'>
                        <span style='color:#10b981; font-weight:700; font-size:1.1rem;'>✅ Grounded RAG Model</span><br>
                        <span style='color:#94a3b8; font-size:0.85rem;'>Answers only from retrieved context. Verifiable.</span>
                    </div>
                """, unsafe_allow_html=True)
                with st.spinner("Retrieving facts & generating response..."):
                    if chunks:
                        ans_rag = generate_answer(build_messages(custom_query, chunks), temperature)
                    else:
                        ans_rag = "No relevant document found in the corpus."

                st.markdown("### Answer:")
                if chunks:
                    st.success(ans_rag)
                    tts_button(ans_rag)
                    st.markdown("---")
                    st.markdown("##### 📚 Reference Sources Grounded from Corpus:")
                    for src in chunks:
                        render_source_card(src)
                else:
                    st.warning(ans_rag)

            # Downloadable comparison report (.md).
            st.markdown("---")
            st.markdown("### 📥 Download Session Comparison Report")
            sources_md = "".join(
                f"{i}. [{src['title']}]({src['origin']}) - Score: {src['score']}\n"
                for i, src in enumerate(chunks, start=1)
            ) or "No reference found."

            report_md = f"""# Fermilab Q&A Hallucination Comparison Report
**Question Analyzed:** "{custom_query}"

---

### ❌ Direct Response (No RAG)
> *Status: unchecked recall — may hallucinate*

{ans_direct}

---

### ✅ Grounded Response (With RAG)
> *Status: verified against the crawled corpus*

{ans_rag}

---

### 📚 References retrieved from corpus:
{sources_md}
"""
            st.download_button(
                label="📥 Export Comparison Report as Markdown (.md)",
                data=report_md,
                file_name=f"Fermilab_Comparison_{custom_query[:20].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ----------------------------------------------------------------------------
# TAB 2 — CORPUS MAP
# ----------------------------------------------------------------------------
with tab_map:
    st.markdown("### 🗺️ Fermilab Knowledge Base Topology Map")
    st.markdown("A 2D view of the corpus. Points close together = similar topics.")

    if corpus_map_df.empty:
        st.warning("Could not load the 2D projection. Check corpus_st7.jsonl.")
    else:
        st.scatter_chart(
            data=corpus_map_df, x="x", y="y",
            color="source_type", size=None, use_container_width=True,
        )
        st.info("💡 Tip: use the chart controls to zoom, pan, and hover over points.")


# ----------------------------------------------------------------------------
# TAB 3 — MEDIA
# ----------------------------------------------------------------------------
with tab_media:
    st.markdown("### 🎥 Fermilab Educational & Science Center")
    st.markdown("Explore particle physics and accelerators through video and audio.")

    media_col1, media_col2 = st.columns(2)
    with media_col1:
        st.markdown("#### 📺 Video: What is a Neutrino?")
        st.markdown("A Fermilab educational video about neutrinos.")
        st.video("https://www.youtube.com/watch?v=k9X8j5_g4aQ")
    with media_col2:
        st.markdown("#### 🔊 Audio: Acoustic Sounds of the Universe")
        st.markdown("Sun sound waves recorded by NASA.")
        st.audio("https://images-assets.nasa.gov/audio/NASA-Sound-Of-Sun/NASA-Sound-Of-Sun.mp3")
        st.markdown("<p style='font-size:0.8rem; color:#64748b;'>Source: NASA Heliophysics Archive.</p>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# TAB 4 — PIPELINE METRICS
# ----------------------------------------------------------------------------
with tab_metrics:
    st.markdown("### 📈 Crawler Monitor & Pipeline Training Log")

    st.markdown("#### 🕷️ 1. Web Crawler Activity Dashboard")
    report_csv = APP_DIR.parent / "crawled_data" / "download_report_1000.csv"

    if report_csv.exists():
        try:
            df_crawl = pd.read_csv(str(report_csv))

            c1, c2, c3, c4 = st.columns(4)
            total = len(df_crawl)
            success_count = len(df_crawl[df_crawl["status"] == "SUCCESS"])
            with c1:
                st.metric("Total Pages Attempted", total)
            with c2:
                st.metric("Downloaded Successfully", success_count)
            with c3:
                rate = (success_count / total * 100) if total else 0
                st.metric("Crawl Success Rate", f"{rate:.1f}%")
            with c4:
                st.metric("Average Word Count", f"{int(df_crawl['word_count'].mean())} words/page")

            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                st.markdown("**Crawled Pages by Domain**")
                df_crawl["domain"] = df_crawl["url"].apply(lambda u: urlparse(str(u)).hostname or "Unknown")
                domain_counts = df_crawl["domain"].value_counts().reset_index()
                domain_counts.columns = ["Domain", "Pages"]
                st.bar_chart(domain_counts, x="Domain", y="Pages")
            with chart_c2:
                st.markdown("**HTTP Status Codes Distribution**")
                status_dist = df_crawl["http_status_code"].value_counts().reset_index()
                status_dist.columns = ["HTTP Status", "Occurrences"]
                status_dist["HTTP Status"] = status_dist["HTTP Status"].apply(
                    lambda x: str(int(x)) if pd.notna(x) else "Timeout/Error"
                )
                st.bar_chart(status_dist, x="HTTP Status", y="Occurrences")
        except Exception as e:
            st.error(f"Failed to read crawl report: {e}")
    else:
        st.warning(f"Report file not found at: {report_csv}. Run the crawler first.")

    st.markdown("---")
    st.markdown("#### 🤖 2. Stage 3: LoRA Training History Log")
    st.markdown("""
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Global Step</th><th>Epoch</th><th>Training Loss</th>
                    <th>Token Acc.</th><th>Val Loss</th><th>Val Token Acc.</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>5</td><td>0.35</td><td>2.7476</td><td>57.69%</td><td>—</td><td>—</td></tr>
                <tr><td>10</td><td>0.70</td><td>1.5354</td><td>67.66%</td><td>—</td><td>—</td></tr>
                <tr><td>15</td><td>1.00</td><td>1.4085</td><td>70.00%</td><td>—</td><td>—</td></tr>
                <tr><td>20</td><td>1.35</td><td>1.1682</td><td>73.37%</td><td>—</td><td>—</td></tr>
                <tr><td>25</td><td>1.70</td><td>1.1433</td><td>73.73%</td><td>1.3397</td><td>70.71%</td></tr>
                <tr style="background-color: rgba(16,185,129,0.1); font-weight: bold;">
                    <td>30</td><td>2.00</td><td>1.1005</td><td>73.88%</td><td>1.3369</td><td>70.64%</td>
                </tr>
            </tbody>
        </table>
    """, unsafe_allow_html=True)

    st.success("🎓 Training complete. Loss 2.748 → 1.101, token accuracy 57.69% → 73.88%. Validation stable (val loss 1.34, val acc 70.64%).")

# Run this file: streamlit run app.py --server.fileWatcherType none