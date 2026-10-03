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
# Example
# ============================================================

EXAMPLE_CONTEXT = (
    "The University of Notre Dame is a private Catholic research "
    "university located in Notre Dame, Indiana, United States. "
    "It was founded in 1842 by Rev. Edward Sorin."
)

EXAMPLE_QUESTION = (
    "Where is the University of Notre Dame located?"
)


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>

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
        background-color: #f1f5f9;
        color: #475569;
        padding: 6px 11px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 13px;
    }

    .hero-title {
        color: #111827;
        font-size: 36px;
        font-weight: 700;
        line-height: 1.2;
        margin: 0;
    }

    .hero-description {
        color: #6b7280;
        font-size: 15px;
        line-height: 1.7;
        margin-top: 10px;
        max-width: 850px;
    }

    .tag {
        display: inline-block;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #475569;
        padding: 5px 9px;
        border-radius: 7px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 5px;
        margin-top: 13px;
    }

    /* Sections */

    .section-title {
        color: #111827;
        font-size: 20px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 12px;
    }

    /* Text Areas */

    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 11px;
        color: #111827;
        font-size: 15px;
        line-height: 1.65;
        padding: 13px;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #64748b;
        box-shadow: 0 0 0 1px #64748b;
    }

    div[data-testid="stTextInput"] input {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 11px;
        color: #111827;
        font-size: 15px;
        padding: 11px 13px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #64748b;
        box-shadow: 0 0 0 1px #64748b;
    }

    /* Buttons */

    div.stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 14px;
    }

    /* Status */

    .status-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 11px 14px;
        margin-bottom: 20px;
        color: #4b5563;
        font-size: 13px;
    }

    /* Answer */

    .answer-card {
        background-color: #ffffff;
        border: 1px solid #dbe1e8;
        border-radius: 14px;
        padding: 23px 25px;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.035);
    }

    .answer-label {
        color: #6b7280;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .answer-text {
        color: #111827;
        font-size: 22px;
        font-weight: 600;
        line-height: 1.55;
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
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .info-text {
        color: #6b7280;
        font-size: 13px;
        line-height: 1.65;
    }

    /* Sidebar */

    .sidebar-title {
        color: #111827;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 3px;
    }

    .sidebar-description {
        color: #6b7280;
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    .sidebar-item {
        margin-bottom: 13px;
    }

    .sidebar-label {
        color: #6b7280;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .sidebar-value {
        color: #111827;
        font-size: 13px;
        margin-top: 2px;
        word-break: break-word;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e5e7eb;
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


# ============================================================
# Load Model
# ============================================================

try:

    engine = load_engine()

except Exception as exc:

    st.error("Failed to load the Question Answering model.")

    with st.expander("Technical Details"):
        st.exception(exc)

    st.stop()


model_info = engine.get_model_info()


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

device_label = str(model_info["device"])

st.markdown(
    f"""
    <div class="status-card">
        Model ready · {html.escape(device_label)} ·
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


context = st.text_area(
    "Context",
    value=EXAMPLE_CONTEXT,
    height=230,
    placeholder="Paste the passage that contains the answer...",
    help="The answer should be contained inside this passage.",
)


question = st.text_input(
    "Question",
    value=EXAMPLE_QUESTION,
    placeholder="Ask a question about the context...",
    help="Ask a question that can be answered directly from the context.",
)


# ============================================================
# Example Button
# ============================================================

example_col1, example_col2 = st.columns([1, 5])

with example_col1:

    if st.button(
        "Load Example",
        use_container_width=True,
    ):

        st.rerun()


# ============================================================
# Inference Button
# ============================================================

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

        st.warning(
            "Please provide a context before running inference."
        )

        st.stop()

    if not question.strip():

        st.warning(
            "Please enter a question before running inference."
        )

        st.stop()

    with st.spinner("Finding the best answer..."):

        try:

            result = engine.answer_question(
                question=question,
                context=context,
            )

        except Exception as exc:

            st.error(
                "An error occurred during inference."
            )

            with st.expander("Technical Details"):
                st.exception(exc)

            st.stop()


    # ========================================================
    # Answer
    # ========================================================

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

        st.stop()


    safe_answer = html.escape(
        str(result["answer"])
    )


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


    # ========================================================
    # Inference Metrics
    # ========================================================

    st.markdown(
        '<div class="section-title">Inference Details</div>',
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:

        st.metric(
            "Features",
            result["num_features"],
        )

    with metric_col2:

        st.metric(
            "Candidates",
            result["num_candidates"],
        )

    with metric_col3:

        st.metric(
            "Selected Feature",
            result["feature_index"],
        )

    with metric_col4:

        st.metric(
            "Latency",
            f"{result['inference_time']:.3f} s",
        )


    st.caption(
        "Latency includes preprocessing, sliding-window construction, "
        "model inference, and answer-span selection."
    )


    # ========================================================
    # Technical Details
    # ========================================================

    with st.expander("Technical Details"):

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:

            st.write(
                f"**Character Start:** "
                f"`{result['start_char']}`"
            )

            st.write(
                f"**Character End:** "
                f"`{result['end_char']}`"
            )

            st.write(
                f"**Selected Feature:** "
                f"`{result['feature_index']}`"
            )

        with detail_col2:

            st.write(
                f"**Total Features:** "
                f"`{result['num_features']}`"
            )

            st.write(
                f"**Generated Candidates:** "
                f"`{result['num_candidates']}`"
            )

            st.write(
                f"**Raw Span Score:** "
                f"`{result['score']:.4f}`"
            )


        st.info(
            "The raw span score is the sum of the predicted "
            "start and end logits. It is not a calibrated "
            "probability and should not be interpreted as "
            "model confidence."
        )


    # ========================================================
    # Input Context
    # ========================================================

    with st.expander("View Input Context"):

        st.text(context)


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
            <div class="info-title">
                1. Tokenization
            </div>
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
            <div class="info-title">
                2. Transformer QA
            </div>
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
            <div class="info-title">
                3. Span Selection
            </div>
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
