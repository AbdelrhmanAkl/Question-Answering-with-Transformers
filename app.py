import html

import streamlit as st

from src.qa_inference import QuestionAnsweringEngine


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Question Answering",
    page_icon="QA",
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


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --accent: #4f46e5;
        --accent-hover: #4338ca;
    }

    .stApp {
        background-color: #f7f8fa;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* Hero */

    .hero {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 30px 32px;
        margin-bottom: 22px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.035);
    }

    .hero-badge {
        display: inline-block;
        background-color: #eef2ff;
        color: var(--accent);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 13px;
    }

    .hero-title {
        color: #111827;
        font-size: 38px;
        font-weight: 700;
        line-height: 1.2;
        margin: 0;
    }

    .hero-description {
        color: #4b5563;
        font-size: 16px;
        line-height: 1.7;
        margin-top: 10px;
        max-width: 850px;
    }

    .tag {
        display: inline-block;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #475569;
        padding: 6px 11px;
        border-radius: 7px;
        font-size: 12.5px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 13px;
    }

    /* Sections */

    .section-title {
        color: #111827;
        font-size: 21px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #4b5563;
        font-size: 15px;
        margin-bottom: 12px;
    }

    /* Inputs */

    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 11px;
        color: #111827;
        font-size: 16px;
        line-height: 1.65;
        padding: 13px;
    }

    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }

    div[data-testid="stTextInput"] input {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 11px;
        color: #111827;
        font-size: 16px;
        padding: 11px 13px;
    }

    /* Buttons */

    div.stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 15px;
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background-color: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #ffffff;
    }

    /* Status */

    .status-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 11px 14px;
        margin-bottom: 20px;
        color: #4b5563;
        font-size: 14px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
        margin-right: 8px;
    }

    /* Answer */

    .answer-card {
        background-color: #ffffff;
        border: 1px solid #c7d2fe;
        border-left: 5px solid var(--accent);
        border-radius: 14px;
        padding: 23px 25px;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(79, 70, 229, 0.07);
    }

    .answer-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .answer-text {
        color: #111827;
        font-size: 24px;
        font-weight: 600;
        line-height: 1.55;
    }

    /* Highlighted context */

    .context-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        color: #374151;
        font-size: 16px;
        line-height: 1.85;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .context-box mark {
        background-color: #fde68a;
        color: #111827;
        padding: 2px 5px;
        border-radius: 5px;
        font-weight: 600;
    }

    /* Metrics */

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 11px;
        padding: 14px;
    }

    /* Information Cards */

    .info-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        min-height: 130px;
    }

    .info-title {
        color: #111827;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .info-text {
        color: #4b5563;
        font-size: 14px;
        line-height: 1.65;
    }

    /* Sidebar */

    .sidebar-title {
        color: #111827;
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .sidebar-description {
        color: #4b5563;
        font-size: 13.5px;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    .sidebar-item {
        margin-bottom: 13px;
    }

    .sidebar-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .sidebar-value {
        color: #111827;
        font-size: 14px;
        margin-top: 2px;
        word-break: break-word;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e5e7eb;
    }

    /* Mobile */

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        .hero {
            padding: 20px 18px;
        }
        .hero-title {
            font-size: 28px;
        }
        .answer-text {
            font-size: 20px;
        }
        .info-card {
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


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            AI Question Answering
        </div>
        <div class="sidebar-description">
            Extractive Question Answering powered by a
            fine-tuned DistilBERT Transformer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Model")

    st.markdown(
        f"""
        <div class="sidebar-item">
            <div class="sidebar-label">Model</div>
            <div class="sidebar-value">{html.escape(MODEL_ID)}</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Architecture</div>
            <div class="sidebar-value">{html.escape(str(model_info["model_class"]))}</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Tokenizer</div>
            <div class="sidebar-value">{html.escape(str(model_info["tokenizer_class"]))}</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Parameters</div>
            <div class="sidebar-value">{model_info["parameters"]:,}</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Device</div>
            <div class="sidebar-value">{html.escape(str(model_info["device"]))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Inference Configuration")

    st.markdown(
        f"""
        <div class="sidebar-item">
            <div class="sidebar-label">Max Sequence Length</div>
            <div class="sidebar-value">{model_info["max_length"]}</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Document Stride</div>
            <div class="sidebar-value">{model_info["doc_stride"]}</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">N-Best Candidates</div>
            <div class="sidebar-value">{model_info["n_best"]}</div>
        </div>
        <div class="sidebar-item">
            <div class="sidebar-label">Max Answer Length</div>
            <div class="sidebar-value">{model_info["max_answer_length"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "The model extracts answer spans directly from the "
        "provided context. No separate generative LLM is used."
    )


# ============================================================
# Hero Section
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            NLP · TRANSFORMER · EXTRACTIVE QA
        </div>
        <div class="hero-title">
            AI Question Answering
        </div>
        <div class="hero-description">
            Ask questions about a passage and let a fine-tuned
            DistilBERT Transformer identify the most relevant
            answer span directly from the provided context.
        </div>
        <div>
            <span class="tag">DistilBERT</span>
            <span class="tag">SQuAD v1.1</span>
            <span class="tag">Transformers</span>
            <span class="tag">Extractive QA</span>
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
        <span class="status-dot"></span>Model ready ·
        {html.escape(str(model_info["device"]))} ·
        Fine-tuned Question Answering model
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Input Section
# ============================================================

st.markdown(
    '<div class="section-title">Ask a Question</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Provide a passage and ask a question about information
        contained within it.
    </div>
    """,
    unsafe_allow_html=True,
)

example_col1, example_col2 = st.columns([3, 1])

with example_col1:

    st.selectbox(
        "Example",
        options=list(EXAMPLES.keys()),
        key="example_choice",
        label_visibility="collapsed",
    )

with example_col2:

    st.button(
        "Load Example",
        use_container_width=True,
        on_click=load_selected_example,
    )


context = st.text_area(
    "Context",
    key="context",
    height=230,
    placeholder="Paste the passage that contains the answer...",
    help="The answer should be contained inside this passage.",
)


question = st.text_input(
    "Question",
    key="question",
    placeholder="Ask a question about the context...",
    help="Ask a question that can be answered directly from the context.",
)


st.write("")

run_inference = st.button(
    "Answer Question",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Inference
# ============================================================

if run_inference:

    if not context.strip():

        st.warning("Please provide a context before running inference.")

    elif not question.strip():

        st.warning("Please enter a question before running inference.")

    else:

        result = None

        with st.spinner("Finding the best answer..."):

            try:

                result = engine.answer_question(
                    question=question,
                    context=context,
                )

            except Exception as exc:

                st.error("An error occurred during inference.")

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
                    "The model could not extract a valid answer "
                    "from the provided context."
                )

            else:

                safe_answer = html.escape(str(result["answer"]))

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-label">
                            Extracted Answer
                        </div>
                        <div class="answer-text">
                            {safe_answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Answer highlighted inside the context
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">Answer in Context</div>',
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
                    '<div class="section-title">Inference Details</div>',
                    unsafe_allow_html=True,
                )

                metric_col1, metric_col2, metric_col3, metric_col4 = (
                    st.columns(4)
                )

                with metric_col1:
                    st.metric("Features", result["num_features"])

                with metric_col2:
                    st.metric("Candidates", result["num_candidates"])

                with metric_col3:
                    st.metric("Selected Feature", result["feature_index"])

                with metric_col4:
                    st.metric("Latency", f"{result['inference_time']:.3f} s")

                st.caption(
                    "Latency includes preprocessing, sliding-window "
                    "construction, model inference, and answer-span selection."
                )

                # ------------------------------------------------
                # Technical details
                # ------------------------------------------------

                with st.expander("Technical Details"):

                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:

                        st.write(
                            f"**Character Start:** `{result['start_char']}`"
                        )
                        st.write(
                            f"**Character End:** `{result['end_char']}`"
                        )
                        st.write(
                            f"**Selected Feature:** `{result['feature_index']}`"
                        )

                    with detail_col2:

                        st.write(
                            f"**Total Features:** `{result['num_features']}`"
                        )
                        st.write(
                            f"**Generated Candidates:** "
                            f"`{result['num_candidates']}`"
                        )
                        st.write(
                            f"**Raw Span Score:** `{result['score']:.4f}`"
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
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        The application uses an extractive Transformer-based
        Question Answering pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)


architecture_col1, architecture_col2, architecture_col3 = st.columns(3)

with architecture_col1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">1. Tokenization</div>
            <div class="info-text">
                The question and context are tokenized while
                preserving offset mappings for answer reconstruction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with architecture_col2:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">2. Transformer QA</div>
            <div class="info-text">
                DistilBERT predicts the start and end positions
                of the answer span inside the context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with architecture_col3:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">3. Span Selection</div>
            <div class="info-text">
                Multiple candidate spans are evaluated and the
                highest-scoring valid answer span is selected.
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
        DistilBERT fine-tuned on SQuAD v1.1
        · Extractive Question Answering
        · Transformers · PyTorch · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
