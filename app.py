import html

import streamlit as st

from src.qa_inference import QuestionAnsweringEngine


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Ask Your Text",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
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

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Nunito+Sans:wght@400;600;700&display=swap');

    :root {
        --paper: #faf6ef;
        --surface: #fffdf9;
        --ink: #2f2a24;
        --muted: #756c60;
        --soft: #a1978a;
        --line: #ebe3d6;
        --accent: #3f7a69;
        --accent-hover: #346756;
        --accent-tint: #e6f0eb;
        --marker: #ffe08a;
        --marker-soft: #fff1c4;
        --font-display: 'Fraunces', Georgia, serif;
        --font-body: 'Nunito Sans', -apple-system, 'Segoe UI', sans-serif;
    }

    /* ---------- Base ---------- */

    html, body, .stApp,
    .stApp p, .stApp label, .stApp li,
    div[data-testid="stMarkdownContainer"],
    textarea, input, button {
        font-family: var(--font-body);
    }

    .stApp {
        background-color: var(--paper);
        background-image: radial-gradient(
            800px 360px at 50% -120px, #fdeccd 0%, rgba(253, 236, 205, 0) 70%
        );
        background-repeat: no-repeat;
        color: var(--ink);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 780px;
        padding-top: 2.6rem;
        padding-bottom: 3rem;
    }

    hr {
        border-color: var(--line) !important;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--line);
    }

    .side-title {
        font-family: var(--font-display);
        font-size: 18px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 4px;
    }

    .side-text {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 14px;
    }

    .side-row {
        padding: 9px 0;
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

    /* ---------- Header ---------- */

    .page-head {
        text-align: center;
        margin-bottom: 26px;
    }

    .page-title {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 42px;
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -0.01em;
        margin: 0;
    }

    .page-sub {
        color: var(--muted);
        font-size: 17px;
        line-height: 1.6;
        margin: 10px auto 0 auto;
        max-width: 520px;
    }

    /* ---------- Steps hint ---------- */

    .field-title {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 18px;
        font-weight: 700;
        margin: 6px 0 2px 0;
    }

    .field-hint {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 8px;
    }

    /* ---------- Inputs ---------- */

    div[data-testid="stTextArea"] label,
    div[data-testid="stTextInput"] label {
        display: none;
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
        box-shadow: 0 0 0 4px rgba(63, 122, 105, 0.14) !important;
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
        min-height: 44px;
        font-weight: 600;
        color: var(--ink);
    }

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
        background: transparent;
    }

    /* ---------- Buttons ---------- */

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 44px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 15px;
        transition: background-color 0.15s ease, box-shadow 0.15s ease;
    }

    button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: var(--surface);
        border: 1px solid var(--line);
        color: var(--muted);
    }

    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--accent);
        color: var(--accent);
        background-color: var(--accent-tint);
    }

    button[kind^="primary"],
    button[data-testid^="stBaseButton-primary"] {
        background-color: var(--accent);
        border: 1px solid var(--accent);
        color: #ffffff;
        min-height: 52px;
        font-size: 16px;
        box-shadow: 0 8px 20px rgba(63, 122, 105, 0.25);
    }

    button[kind^="primary"]:hover,
    button[data-testid^="stBaseButton-primary"]:hover {
        background-color: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #ffffff;
    }

    /* ---------- Result ---------- */

    .result-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 28px 30px;
        margin-top: 8px;
        box-shadow: 0 14px 36px rgba(95, 70, 30, 0.08);
    }

    .result-label {
        color: var(--muted);
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .result-answer {
        font-family: var(--font-display);
        color: var(--ink);
        font-size: 34px;
        font-weight: 700;
        line-height: 1.35;
        background-image: linear-gradient(transparent 60%, var(--marker) 60%);
        padding: 0 4px;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
    }

    .result-divider {
        height: 1px;
        background: var(--line);
        margin: 22px 0 18px 0;
    }

    .result-context {
        color: #4a433a;
        font-size: 16.5px;
        line-height: 1.95;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .result-context mark {
        background: var(--marker);
        color: var(--ink);
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 700;
    }

    .result-meta {
        margin-top: 16px;
        color: var(--soft);
        font-size: 13px;
    }

    /* ---------- Expanders ---------- */

    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        margin-top: 14px;
    }

    div[data-testid="stExpander"] summary p {
        font-weight: 700;
        color: var(--ink);
    }

    .step-line {
        color: var(--muted);
        font-size: 14.5px;
        line-height: 1.75;
        margin-bottom: 8px;
    }

    .step-line b {
        color: var(--ink);
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: var(--soft);
        font-size: 13px;
        margin-top: 36px;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.2rem;
        }
        .page-title {
            font-size: 32px;
        }
        .result-card {
            padding: 22px 20px;
        }
        .result-answer {
            font-size: 26px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource(show_spinner="Getting things ready...")
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

    st.error("The model could not be loaded. Please refresh the page.")

    with st.expander("Technical details"):
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


def sidebar_row(label: str, value) -> str:
    return (
        '<div class="side-row">'
        f'<div class="side-label">{html.escape(label)}</div>'
        f'<div class="side-value">{html.escape(str(value))}</div>'
        "</div>"
    )


# ============================================================
# Sidebar (model info, hidden by default)
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="side-title">About this app</div>
        <div class="side-text">
            The model finds the answer inside your passage.
            It does not make up new text.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        sidebar_row("Model", MODEL_ID)
        + sidebar_row("Architecture", model_info["model_class"])
        + sidebar_row("Parameters", f'{model_info["parameters"]:,}')
        + sidebar_row("Device", model_info["device"])
        + sidebar_row("Max sequence length", model_info["max_length"])
        + sidebar_row("Document stride", model_info["doc_stride"])
        + sidebar_row("N-best candidates", model_info["n_best"])
        + sidebar_row("Max answer length", model_info["max_answer_length"]),
        unsafe_allow_html=True,
    )


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="page-head">
        <div class="page-title">Ask your text</div>
        <div class="page-sub">
            Paste a passage, ask a question, and get the answer
            highlighted right where it appears.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Examples
# ============================================================

ex_col1, ex_col2 = st.columns([4, 1])

with ex_col1:

    st.selectbox(
        "Try an example",
        options=list(EXAMPLES.keys()),
        key="example_choice",
        label_visibility="collapsed",
        on_change=load_selected_example,
    )

with ex_col2:

    st.button("Clear", use_container_width=True, on_click=clear_inputs)


# ============================================================
# Input Form (press Enter in the question box to submit)
# ============================================================

with st.form("qa_form"):

    st.markdown(
        """
        <div class="field-title">1. Your passage</div>
        <div class="field-hint">Paste the text that contains the answer.</div>
        """,
        unsafe_allow_html=True,
    )

    context = st.text_area(
        "Passage",
        key="context",
        height=200,
        placeholder="Paste or type your text here...",
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="field-title">2. Your question</div>
        <div class="field-hint">Ask something the passage can answer.</div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Question",
        key="question",
        placeholder="Type your question here...",
        label_visibility="collapsed",
    )

    run_inference = st.form_submit_button(
        "Find the answer",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# Inference
# ============================================================

if run_inference:

    if not context.strip():

        st.warning("Add a passage first, then try again.")

    elif not question.strip():

        st.warning("Type a question first, then try again.")

    else:

        result = None

        with st.spinner("Searching your text..."):

            try:

                result = engine.answer_question(
                    question=question,
                    context=context,
                )

            except Exception as exc:

                st.error("Something went wrong while searching. Please try again.")

                with st.expander("Technical details"):
                    st.exception(exc)

        if result is not None:

            st.write("")

            if not result["answer"]:

                st.warning(
                    "No answer was found in this passage. "
                    "Try rephrasing the question or adding more text."
                )

            else:

                highlighted = highlight_answer(
                    context,
                    result.get("start_char"),
                    result.get("end_char"),
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Answer</div>
                        <div class="result-answer">{html.escape(str(result["answer"]))}</div>
                        <div class="result-divider"></div>
                        <div class="result-label">Found in your text</div>
                        <div class="result-context">{highlighted}</div>
                        <div class="result-meta">
                            Answered in {result["inference_time"]:.2f} seconds
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("Technical details"):

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Character start:** `{result['start_char']}`")
                        st.write(f"**Character end:** `{result['end_char']}`")
                        st.write(f"**Raw span score:** `{result['score']:.4f}`")

                    with col2:
                        st.write(f"**Features:** `{result['num_features']}`")
                        st.write(f"**Candidates:** `{result['num_candidates']}`")
                        st.write(f"**Selected feature:** `{result['feature_index']}`")

                    st.caption(
                        "The raw span score is the sum of the start and end "
                        "logits. It is not a probability and should not be "
                        "read as model confidence."
                    )


# ============================================================
# How It Works
# ============================================================

with st.expander("How does it work?"):

    st.markdown(
        """
        <div class="step-line">
            <b>1. Read.</b> Your question and passage are split into
            tokens, keeping track of where each one sits in the text.
        </div>
        <div class="step-line">
            <b>2. Predict.</b> DistilBERT predicts where the answer
            starts and ends inside the passage.
        </div>
        <div class="step-line">
            <b>3. Pick.</b> Several candidate answers are scored and the
            best valid one is shown.
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
        DistilBERT fine-tuned on SQuAD v1.1
    </div>
    """,
    unsafe_allow_html=True,
)
