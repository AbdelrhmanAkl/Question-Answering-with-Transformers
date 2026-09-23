import streamlit as st

from src.qa_inference import QuestionAnsweringEngine


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Question Answering with Transformers",
    page_icon="QA",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Model Configuration
# ============================================================

MODEL_ID = "AbdelrahmanAkl/distilbert-squad-qa"


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
        .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem 0 1rem 0;
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            line-height: 1.6;
            opacity: 0.75;
            max-width: 850px;
        }

        .answer-box {
            padding: 1.35rem 1.5rem;
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 14px;
            margin-top: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .answer-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.65;
            margin-bottom: 0.55rem;
        }

        .answer-text {
            font-size: 1.45rem;
            line-height: 1.55;
            font-weight: 600;
        }

        .section-description {
            opacity: 0.7;
            margin-top: -0.4rem;
            margin-bottom: 1rem;
        }

        .info-box {
            padding: 1rem 1.15rem;
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 10px;
            margin-top: 0.5rem;
        }

        .footer {
            text-align: center;
            opacity: 0.55;
            font-size: 0.85rem;
            padding-top: 0.75rem;
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
        max_length=384,
        doc_stride=128,
        n_best=20,
        max_answer_length=30,
    )


# ============================================================
# Load Model
# ============================================================

try:
    engine = load_engine()

except Exception as exc:
    st.error("Failed to load the Question Answering model.")

    with st.expander("Technical details"):
        st.exception(exc)

    st.stop()


model_info = engine.get_model_info()


# ============================================================
# Hero Section
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Question Answering with Transformers
        </div>
        <div class="hero-subtitle">
            Extractive Question Answering powered by DistilBERT,
            fine-tuned on SQuAD v1.1. The model identifies answer
            spans directly from the provided context.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Model Information")

    st.write(
        f"**Model**  \n"
        f"`{MODEL_ID}`"
    )

    st.write(
        f"**Architecture**  \n"
        f"`{model_info['model_class']}`"
    )

    st.write(
        f"**Tokenizer**  \n"
        f"`{model_info['tokenizer_class']}`"
    )

    st.write(
        f"**Parameters**  \n"
        f"`{model_info['parameters']:,}`"
    )

    st.write(
        f"**Device**  \n"
        f"`{model_info['device']}`"
    )

    st.divider()

    st.header("Inference Configuration")

    st.write(
        f"**Max sequence length**  \n"
        f"`{model_info['max_length']}`"
    )

    st.write(
        f"**Document stride**  \n"
        f"`{model_info['doc_stride']}`"
    )

    st.write(
        f"**N-best candidates**  \n"
        f"`{model_info['n_best']}`"
    )

    st.write(
        f"**Max answer length**  \n"
        f"`{model_info['max_answer_length']}`"
    )

    st.divider()

    st.caption(
        "Answers are extracted directly from the supplied context. "
        "The application does not generate answers with a separate "
        "generative LLM."
    )


# ============================================================
# Input Section
# ============================================================

st.subheader("Ask a Question")

st.markdown(
    '<div class="section-description">'
    "Provide a passage and ask a question about it."
    "</div>",
    unsafe_allow_html=True,
)

context = st.text_area(
    "Context",
    value=EXAMPLE_CONTEXT,
    height=240,
    placeholder="Paste the passage that contains the answer...",
    help="The model extracts the answer directly from this passage.",
)

question = st.text_input(
    "Question",
    value=EXAMPLE_QUESTION,
    placeholder="Ask a question about the context...",
    help="The answer should be contained in the provided context.",
)


# ============================================================
# Example Controls
# ============================================================

example_col1, example_col2 = st.columns([1, 4])

with example_col1:
    use_example = st.button(
        "Load Example",
        use_container_width=True,
    )

if use_example:
    st.session_state["example_loaded"] = True


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

    with st.spinner("Running extractive QA inference..."):

        try:
            result = engine.answer_question(
                question=question,
                context=context,
            )

        except Exception as exc:
            st.error("An error occurred during inference.")

            with st.expander("Technical details"):
                st.exception(exc)

            st.stop()

    # ========================================================
    # Answer
    # ========================================================

    st.divider()
    st.subheader("Answer")

    if not result["answer"]:
        st.warning(
            "The model could not extract a valid answer span "
            "from the provided context."
        )
        st.stop()

    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-label">Extracted Answer</div>
            <div class="answer-text">{result["answer"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # Metrics
    # ========================================================

    st.subheader("Inference Details")

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
                f"**Character start:** "
                f"`{result['start_char']}`"
            )

            st.write(
                f"**Character end:** "
                f"`{result['end_char']}`"
            )

            st.write(
                f"**Selected feature:** "
                f"`{result['feature_index']}`"
            )

        with detail_col2:
            st.write(
                f"**Total features:** "
                f"`{result['num_features']}`"
            )

            st.write(
                f"**Generated candidates:** "
                f"`{result['num_candidates']}`"
            )

            st.write(
                f"**Raw span score:** "
                f"`{result['score']:.4f}`"
            )

        st.info(
            "The raw span score is the sum of the predicted start "
            "and end logits. It is not a calibrated probability "
            "and should not be interpreted as model confidence."
        )

    # ========================================================
    # Input Context
    # ========================================================

    with st.expander("View Input Context"):
        st.text(context)


# ============================================================
# Architecture Section
# ============================================================

st.divider()

st.subheader("How It Works")

architecture_col1, architecture_col2, architecture_col3 = st.columns(3)

with architecture_col1:
    st.markdown(
        """
        **1. Tokenization**

        The question and context are tokenized separately,
        with offset mappings preserved for answer reconstruction.
        """
    )

with architecture_col2:
    st.markdown(
        """
        **2. Transformer QA**

        DistilBERT predicts start and end positions for the
        answer span inside the context.
        """
    )

with architecture_col3:
    st.markdown(
        """
        **3. Span Selection**

        Candidate spans are generated and the highest-scoring
        valid answer span is selected.
        """
    )


# ============================================================
# Footer
# ============================================================

st.divider()

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
