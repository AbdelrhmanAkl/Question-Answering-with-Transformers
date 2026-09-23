from qa_inference import QuestionAnsweringEngine


MODEL_ID = "AbdelrahmanAkl/distilbert-squad-qa"


def print_result(
    test_name: str,
    expected: str,
    result: dict,
) -> None:
    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)
    print(f"Expected:       {expected}")
    print(f"Predicted:      {result['answer']}")
    print(f"Score:          {result['score']}")
    print(f"Features:       {result['num_features']}")
    print(f"Candidates:     {result['num_candidates']}")
    print(f"Inference time: {result['inference_time']:.4f} sec")
    print(f"Feature index:  {result['feature_index']}")


def main():
    print("=" * 70)
    print("QUESTION ANSWERING INFERENCE TEST SUITE")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")

    # ---------------------------------------------------------
    # Initialize inference engine
    # ---------------------------------------------------------

    engine = QuestionAnsweringEngine(
        model_name_or_path=MODEL_ID,
        max_length=384,
        doc_stride=128,
        n_best=20,
        max_answer_length=30,
    )

    model_info = engine.get_model_info()

    print("\nMODEL INFORMATION")
    print("-" * 70)

    for key, value in model_info.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # TEST 1: Short Context
    # ---------------------------------------------------------

    context_1 = (
        "The University of Notre Dame is located in South Bend, Indiana. "
        "It was founded in 1842 by Edward Sorin."
    )

    question_1 = (
        "Where is the University of Notre Dame located?"
    )

    expected_1 = "South Bend, Indiana"

    result_1 = engine.answer_question(
        question=question_1,
        context=context_1,
    )

    print_result(
        "TEST 1 - SHORT CONTEXT",
        expected_1,
        result_1,
    )

    assert result_1["answer"], (
        "Test 1 failed: empty answer."
    )

    assert "south bend" in result_1["answer"].lower(), (
        f"Test 1 failed: unexpected answer: "
        f"{result_1['answer']}"
    )

    assert result_1["num_features"] == 1, (
        "Test 1 failed: short context should produce "
        "exactly one feature."
    )

    print("STATUS: PASS")

    # ---------------------------------------------------------
    # TEST 2: Long Context / Sliding Windows
    # ---------------------------------------------------------

    base_context = (
        "The University of Notre Dame is located in South Bend, Indiana. "
        "It was founded in 1842 by Edward Sorin. "
    )

    long_context = base_context * 90

    long_context += (
        "The university was founded by Edward Sorin, "
        "a French priest and missionary."
    )

    question_2 = (
        "Who founded the University of Notre Dame?"
    )

    expected_2 = "Edward Sorin"

    result_2 = engine.answer_question(
        question=question_2,
        context=long_context,
    )

    print_result(
        "TEST 2 - LONG CONTEXT / SLIDING WINDOWS",
        expected_2,
        result_2,
    )

    assert result_2["answer"], (
        "Test 2 failed: empty answer."
    )

    assert "edward sorin" in result_2["answer"].lower(), (
        f"Test 2 failed: unexpected answer: "
        f"{result_2['answer']}"
    )

    assert result_2["num_features"] > 1, (
        "Test 2 failed: long context did not produce "
        "multiple sliding-window features."
    )

    print("STATUS: PASS")

    # ---------------------------------------------------------
    # TEST 3: SQuAD-style Real Example
    # ---------------------------------------------------------

    context_3 = (
        "Super Bowl 50 was an American football game to determine "
        "the champion of the National Football League (NFL) for the "
        "2015 season. The American Football Conference (AFC) champion "
        "Denver Broncos defeated the National Football Conference (NFC) "
        "champion Carolina Panthers 24–10 to earn their third Super Bowl "
        "title. The game was played on February 7, 2016, at Levi's Stadium "
        "in the San Francisco Bay Area at Santa Clara, California."
    )

    question_3 = (
        "Which NFL team represented the AFC at Super Bowl 50?"
    )

    expected_3 = "Denver Broncos"

    result_3 = engine.answer_question(
        question=question_3,
        context=context_3,
    )

    print_result(
        "TEST 3 - SQUAD-STYLE EXAMPLE",
        expected_3,
        result_3,
    )

    assert result_3["answer"], (
        "Test 3 failed: empty answer."
    )

    assert "denver broncos" in result_3["answer"].lower(), (
        f"Test 3 failed: unexpected answer: "
        f"{result_3['answer']}"
    )

    assert result_3["num_features"] == 1, (
        "Test 3 failed: SQuAD-style context should "
        "fit into a single feature."
    )

    print("STATUS: PASS")

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL INFERENCE TESTS PASSED")
    print("=" * 70)
    print("Short-context inference: PASS")
    print("Long-context sliding windows: PASS")
    print("SQuAD-style inference: PASS")
    print("Model loading: PASS")
    print("CPU inference: PASS")
    print("Hugging Face model loading: PASS")
    print("Production inference backend: READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
