import html

import streamlit as st

from src.qa_inference import QuestionAnsweringEngine


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Question Answering",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Model Configuration
# ============================================================

MODEL_ID = "AbdelrahmanAkl/distilbert-squad-qa"

MAX_LENGTH = 384
DOC_STRIDE = 128
N_BEST = 20
MAX_ANSWER_LENGTH = 30


# ============================================================
# Examples
# ============================================================

EXAMPLES = {
    "Notre Dame": {
        "context": (
            "The University of Notre Dame is a private Catholic research "
            "university located in Notre Dame, Indiana, United States. "
            "It was founded in 1842 by Rev. Edward Sorin."
        ),
        "question": "Where is the University of Notre Dame located?",
    },
    "Eiffel Tower": {
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the "
            "Champ de Mars in Paris, France. It is named after the "
            "engineer Gustave Eiffel, whose company designed and built "
            "the tower. It was completed in 1889 for the World's Fair."
        ),
        "question": "Who designed and built the Eiffel Tower?",
    },
    "Amazon River": {
        "context": (
            "The Amazon River in South America is the largest river by "
            "discharge volume of water in the world. It flows through "
            "Peru, Colombia and Brazil before reaching the Atlantic Ocean. "
            "The river system is surrounded by the Amazon rainforest."
        ),
        "question": "Which ocean does the Amazon River flow into?",
    },
    "Photosynthesis": {
        "context": (
            "Photosynthesis is a process used by plants and other "
            "organisms to convert light energy into chemical energy. "
            "This chemical energy is stored in carbohydrate molecules, "
            "such as sugars, which are synthesized from carbon dioxide "
            "and water."
        ),
        "question": "What is photosynthesis used to convert?",
    },
}

DEFAULT_EXAMPLE = "Notre Dame"


# ============================================================
# Session State
# ============================================================

if "context" not in st.session_state:
    st.session_state["context"] = EXAMPLES[DEFAULT_EXAMPLE]["context"]

if "question" not in st.session_state:
    st.session_state["question"] = EXAMPLES[DEFAULT_EXAMPLE]["question"]


def load_selected_example() -> None:
    example = EXAMPLES[st.session_state["example_choice"]]
    st.session_state["context"] = example["context"]
    st.session_state["question"] = example["question"]


def clear_inputs() -> None:
    st.session_state["context"] = ""
    st.session_state["question"] = ""


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700;12..96,800&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --ink: #0f1b2d;
        --muted: #5b6b7f;
        --soft: #8a98aa;
        --line: #e3eaf2;
        --mist: #f5f8fc;
        --surface: #ffffff;
        --accent: #4361ee;
        --accent-hover: #3550d6;
        --accent-tint: #eaf0ff;
        --mint: #12b886;
        --mint-tint: #e6faf3;
        --marker: #ffe97a;
        --marker-soft: #fff6bf;
        --font-display: 'Bricolage Grotesque', 'Manrope', sans-serif;
        --font-body: 'Manrope', -apple-system, 'Segoe UI', sans-serif;
    }

    /* ---------- Base ---------- */

    html, body, .stApp,
    .stApp p, .stApp label, .stApp li,
    div[data-testid="stMarkdownContainer"],
    textarea, input, button {
        font-family: var(--font-body);
    }

    .stApp {
        background-color: var(--mist);
        background-image:
            radial-gradient(900px 420px at 85% -80px, #e4ecff 0%, rgba(228, 236, 255, 0) 70%),
            radial-gradient(700px 380px at -10% 0px, #e3f8f1 0%, rgba(227, 248, 241, 0) 70%);
        background-repeat: no-repeat;
        color: var(--ink);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    hr {
        border-color: var(--line) !important;
        margin: 1.8rem 0 !important;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background-color: #fbfcfe;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .brand-mark {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: var(--accent);
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 16px;
        box-shadow: 0 6px 16px rgba(67, 97, 238, 0.28);
    }

    .brand-name {
        font-family: var(--font-display);
        font-size: 18px;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.2;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 22px;
    }

    .side-heading {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 15px;
        font-weight: 700;
        margin: 4px 0 10px 0;
    }

    .side-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 4px 14px;
        margin-bottom: 18px;
    }

    .side-row {
        padding: 10px 0;
        border-bottom: 1px solid var(--line);
    }

    .side-row:last-child {
        border-bottom: none;
    }

    .side-label {
        color: var(--soft);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 2px;
    }

    .side-value {
        color: var(--ink);
        font-size: 13.5px;
        font-weight: 600;
        word-break: break-word;
    }

    .side-note {
        color: var(--muted);
        font-size: 12.5px;
        line-height: 1.65;
        background: var(--accent-tint);
        border-radius: 12px;
        padding: 12px 14px;
    }

    /* ---------- Hero ---------- */

    .hero {
        display: grid;
        grid-template-columns: 1.25fr 1fr;
        gap: 32px;
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 38px 40px;
        margin-bottom: 18px;
        box-shadow: 0 18px 48px rgba(31, 52, 99, 0.07);
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--accent-tint);
        color: var(--accent);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .hero-pill i {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent);
        display: inline-block;
    }

    .hero-title {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 46px;
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .hero-description {
        color: var(--muted);
        font-size: 16.5px;
        line-height: 1.7;
        margin-top: 14px;
        max-width: 520px;
    }

    .tags {
        margin-top: 20px;
    }

    .tag {
        display: inline-block;
        background: var(--mist);
        border: 1px solid var(--line);
        color: #3d4c63;
        padding: 6px 13px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 0 6px 8px 0;
    }

    /* Hero demo card: shows what the app does */

    .demo {
        background: var(--mist);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 22px;
    }

    .demo-q {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px 14px 14px 4px;
        padding: 12px 15px;
        color: var(--ink);
        font-size: 14px;
        font-weight: 600;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    .demo-ctx {
        color: #3d4c63;
        font-size: 14px;
        line-height: 1.85;
        padding: 4px 4px 0 4px;
    }

    .demo-ctx mark {
        background: var(--marker);
        color: var(--ink);
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 700;
    }

    .demo-foot {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        color: var(--muted);
        font-size: 12.5px;
        font-weight: 600;
    }

    .demo-foot i {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--mint);
        display: inline-block;
    }

    /* ---------- Status ---------- */

    .status-card {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: var(--mint-tint);
        border: 1px solid #c4eddf;
        border-radius: 999px;
        padding: 8px 16px;
        margin-bottom: 8px;
        color: #0b7d5c;
        font-size: 13.5px;
        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--mint);
        box-shadow: 0 0 0 4px rgba(18, 184, 134, 0.18);
        display: inline-block;
    }

    /* ---------- Sections ---------- */

    .section-title {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-top: 22px;
        margin-bottom: 4px;
    }

    .section-description {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    /* ---------- Inputs ---------- */

    div[data-testid="stTextArea"] label p,
    div[data-testid="stTextInput"] label p {
        color: var(--ink);
        font-weight: 700;
        font-size: 14px;
    }

    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(67, 97, 238, 0.12) !important;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: transparent !important;
        color: var(--ink);
        font-size: 16px;
        line-height: 1.7;
        padding: 14px 16px;
    }

    div[data-testid="stTextInput"] input {
        background-color: transparent !important;
        color: var(--ink);
        font-size: 16px;
        padding: 13px 16px;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        min-height: 46px;
        font-weight: 600;
        color: var(--ink);
    }

    /* ---------- Buttons ---------- */

    div.stButton > button {
        min-height: 46px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 15px;
        transition: background-color 0.15s ease, box-shadow 0.15s ease,
                    transform 0.15s ease;
    }

    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"] {
        background-color: var(--surface);
        border: 1px solid var(--line);
        color: var(--ink);
    }

    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--accent);
        color: var(--accent);
        background-color: var(--accent-tint);
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background-color: var(--accent);
        border: 1px solid var(--accent);
        color: #ffffff;
        min-height: 52px;
        font-size: 16px;
        box-shadow: 0 10px 24px rgba(67, 97, 238, 0.28);
    }

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #ffffff;
        transform: translateY(-1px);
    }

    /* ---------- Answer ---------- */

    .answer-card {
        display: flex;
        align-items: center;
        gap: 18px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 26px 28px;
        margin-top: 10px;
        margin-bottom: 22px;
        box-shadow: 0 16px 40px rgba(31, 52, 99, 0.07);
    }

    .answer-icon {
        flex: 0 0 auto;
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: var(--mint-tint);
        color: var(--mint);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
    }

    .answer-label {
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .answer-text {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 30px;
        font-weight: 700;
        line-height: 1.3;
        letter-spacing: -0.01em;
        background-image: linear-gradient(transparent 62%, var(--marker) 62%);
        display: inline;
        padding: 0 4px;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
    }

    /* ---------- Highlighted context ---------- */

    .context-box {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px 26px;
        color: #33425a;
        font-size: 16.5px;
        line-height: 1.95;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .context-box mark {
        background: var(--marker);
        color: var(--ink);
        padding: 3px 7px;
        border-radius: 7px;
        font-weight: 700;
    }

    /* ---------- Metrics ---------- */

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }

    .metric {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .metric-value {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 30px;
        font-weight: 700;
        line-height: 1.1;
    }

    .metric-value small {
        font-size: 16px;
        color: var(--soft);
        font-weight: 600;
        margin-left: 3px;
    }

    .metric.is-accent {
        background: var(--accent-tint);
        border-color: #d3deff;
    }

    .metric.is-accent .metric-value {
        color: var(--accent);
    }

    .caption {
        color: var(--soft);
        font-size: 13px;
        margin-top: 12px;
        line-height: 1.6;
    }

    /* ---------- Expander ---------- */

    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        margin-top: 10px;
    }

    div[data-testid="stExpander"] summary p {
        font-weight: 700;
        color: var(--ink);
    }

    /* ---------- How it works ---------- */

    .step-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px;
        min-height: 164px;
    }

    .step-num {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: var(--accent-tint);
        color: var(--accent);
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 14px;
    }

    .step-title {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .step-text {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: var(--soft);
        font-size: 13px;
        margin-top: 44px;
        padding-top: 18px;
        border-top: 1px solid var(--line);
        line-height: 1.8;
    }

    /* ---------- Mobile ---------- */

    @media (max-width: 900px) {
        .hero {
            grid-template-columns: 1fr;
            padding: 26px 22px;
            gap: 22px;
        }
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        .hero-title {
            font-size: 34px;
        }
        .answer-card {
            padding: 20px 18px;
        }
        .answer-text {
            font-size: 24px;
        }
        .step-card {
            min-height: auto;
            margin-bottom: 10px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource(show_spinner="Loading Question Answering model...")
def load_engine() -> QuestionAnsweringEngine:

    return QuestionAnsweringEngine(
        model_name_or_path=MODEL_ID,
        max_length=MAX_LENGTH,
        doc_stride=DOC_STRIDE,
        n_best=N_BEST,
        max_answer_length=MAX_ANSWER_LENGTH,
    )


try:

    engine = load_engine()

except Exception as exc:

    st.error("Failed to load the Question Answering model.")

    with st.expander("Technical Details"):
        st.exception(exc)

    st.stop()


model_info = engine.get_model_info()


# ============================================================
# Helpers
# ============================================================

def highlight_answer(text: str, start: int, end: int) -> str:
    """Return escaped HTML of `text` with the [start, end) span highlighted."""

    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end > len(text)
        or start >= end
    ):
        return html.escape(text)

    return (
        html.escape(text[:start])
        + "<mark>"
        + html.escape(text[start:end])
        + "</mark>"
        + html.escape(text[end:])
    )


def sidebar_row(label: str, value: str) -> str:
    return (
        '<div class="side-row">'
        f'<div class="side-label">{html.escape(label)}</div>'
        f'<div class="side-value">{html.escape(str(value))}</div>'
        "</div>"
    )


def metric_card(label: str, value: str, unit: str = "", accent: bool = False) -> str:
    extra = " is-accent" if accent else ""
    unit_html = f"<small>{html.escape(unit)}</small>" if unit else ""
    return (
        f'<div class="metric{extra}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(str(value))}{unit_html}</div>'
        "</div>"
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">QA</div>
            <div class="brand-name">AI Question<br>Answering</div>
        </div>
        <div class="brand-sub">
            Finds the answer to your question inside any passage,
            using a fine-tuned DistilBERT model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-heading">Model</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="side-card">'
        + sidebar_row("Model", MODEL_ID)
        + sidebar_row("Architecture", model_info["model_class"])
        + sidebar_row("Tokenizer", model_info["tokenizer_class"])
        + sidebar_row("Parameters", f'{model_info["parameters"]:,}')
        + sidebar_row("Device", model_info["device"])
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-heading">Inference settings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-card">'
        + sidebar_row("Max sequence length", model_info["max_length"])
        + sidebar_row("Document stride", model_info["doc_stride"])
        + sidebar_row("N-best candidates", model_info["n_best"])
        + sidebar_row("Max answer length", model_info["max_answer_length"])
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-note">
            The model extracts answer spans directly from the provided
            context. No separate generative LLM is used.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Hero Section
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div>
            <div class="hero-pill"><i></i>Extractive question answering</div>
            <div class="hero-title">Ask your text,<br>get the exact answer.</div>
            <div class="hero-description">
                Paste a passage, ask a question, and a fine-tuned DistilBERT
                Transformer highlights the answer right where it appears.
            </div>
            <div class="tags">
                <span class="tag">DistilBERT</span>
                <span class="tag">SQuAD v1.1</span>
                <span class="tag">Transformers</span>
                <span class="tag">PyTorch</span>
            </div>
        </div>
        <div class="demo">
            <div class="demo-q">Where was the Eiffel Tower completed?</div>
            <div class="demo-ctx">
                The Eiffel Tower is a wrought-iron lattice tower on the
                Champ de Mars in <mark>Paris, France</mark>. It was completed
                in 1889 for the World's Fair.
            </div>
            <div class="demo-foot"><i></i>Answer found in the passage</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Status
# ============================================================

st.markdown(
    f"""
    <div class="status-card">
        <span class="status-dot"></span>
        Model ready on {html.escape(str(model_info["device"]))}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Input Section
# ============================================================

st.markdown(
    '<div class="section-title">Ask a question</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Pick an example or write your own passage and question.
    </div>
    """,
    unsafe_allow_html=True,
)

example_col1, example_col2, example_col3 = st.columns([3, 1, 1])

with example_col1:

    st.selectbox(
        "Example",
        options=list(EXAMPLES.keys()),
        key="example_choice",
        label_visibility="collapsed",
        on_change=load_selected_example,
    )

with example_col2:

    st.button(
        "Load example",
        use_container_width=True,
        on_click=load_selected_example,
    )

with example_col3:

    st.button(
        "Clear",
        use_container_width=True,
        on_click=clear_inputs,
    )


context = st.text_area(
    "Passage",
    key="context",
    height=230,
    placeholder="Paste the passage that contains the answer...",
    help="The answer should be contained inside this passage.",
)


question = st.text_input(
    "Question",
    key="question",
    placeholder="Ask a question about the passage...",
    help="Ask a question that can be answered directly from the passage.",
)


st.write("")

run_inference = st.button(
    "Find the answer",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Inference
# ============================================================

if run_inference:

    if not context.strip():

        st.warning("Add a passage first, then run again.")

    elif not question.strip():

        st.warning("Type a question first, then run again.")

    else:

        result = None

        with st.spinner("Finding the best answer..."):

            try:

                result = engine.answer_question(
                    question=question,
                    context=context,
                )

            except Exception as exc:

                st.error("Something went wrong while answering.")

                with st.expander("Technical Details"):
                    st.exception(exc)

        if result is not None:

            st.divider()

            st.markdown(
                '<div class="section-title">Answer</div>',
                unsafe_allow_html=True,
            )

            if not result["answer"]:

                st.warning(
                    "No answer was found in this passage. "
                    "Try rephrasing the question or adding more context."
                )

            else:

                safe_answer = html.escape(str(result["answer"]))

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-icon">✓</div>
                        <div>
                            <div class="answer-label">Extracted answer</div>
                            <div class="answer-text">{safe_answer}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Answer highlighted inside the context
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">Answer in context</div>',
                    unsafe_allow_html=True,
                )

                highlighted = highlight_answer(
                    context,
                    result.get("start_char"),
                    result.get("end_char"),
                )

                st.markdown(
                    f'<div class="context-box">{highlighted}</div>',
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Inference metrics
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">Inference details</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="metric-grid">'
                    + metric_card("Features", result["num_features"])
                    + metric_card("Candidates", result["num_candidates"])
                    + metric_card("Selected feature", result["feature_index"])
                    + metric_card(
                        "Latency",
                        f"{result['inference_time']:.3f}",
                        unit="s",
                        accent=True,
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="caption">
                        Latency includes preprocessing, sliding-window
                        construction, model inference, and answer-span selection.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Technical details
                # ------------------------------------------------

                with st.expander("Technical Details"):

                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:

                        st.write(
                            f"**Character start:** `{result['start_char']}`"
                        )
                        st.write(
                            f"**Character end:** `{result['end_char']}`"
                        )
                        st.write(
                            f"**Selected feature:** `{result['feature_index']}`"
                        )

                    with detail_col2:

                        st.write(
                            f"**Total features:** `{result['num_features']}`"
                        )
                        st.write(
                            f"**Generated candidates:** "
                            f"`{result['num_candidates']}`"
                        )
                        st.write(
                            f"**Raw span score:** `{result['score']:.4f}`"
                        )

                    st.info(
                        "The raw span score is the sum of the predicted "
                        "start and end logits. It is not a calibrated "
                        "probability and should not be interpreted as "
                        "model confidence."
                    )


# ============================================================
# How It Works
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">How it works</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Three steps turn your passage and question into an answer.
    </div>
    """,
    unsafe_allow_html=True,
)


architecture_col1, architecture_col2, architecture_col3 = st.columns(3)

with architecture_col1:

    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">Tokenization</div>
            <div class="step-text">
                The question and passage are split into tokens, keeping
                offset mappings so the answer can be mapped back to the text.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with architecture_col2:

    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">Transformer QA</div>
            <div class="step-text">
                DistilBERT predicts where the answer starts and ends
                inside the passage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with architecture_col3:

    st.markdown(
        """
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">Span selection</div>
            <div class="step-text">
                Multiple candidate spans are scored and the best valid
                one is returned as the answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        DistilBERT fine-tuned on SQuAD v1.1<br>
        Built with Transformers, PyTorch and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
