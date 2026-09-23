from pathlib import Path
from typing import Dict, List
import time

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class QuestionAnsweringEngine:
    """
    Production inference engine for extractive Question Answering.

    Features:
    - DistilBERT Question Answering
    - Explicit long-context sliding windows
    - Character-level answer span reconstruction
    - Dynamic padding
    - CPU/GPU inference
    - Top-N start/end span candidate generation
    - Hugging Face Hub model loading
    - Optional local model fallback
    """

    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        max_length: int = 384,
        doc_stride: int = 128,
        n_best: int = 20,
        max_answer_length: int = 30,
        local_files_only: bool = False,
        model_dir: str | Path | None = None,
    ):
        # Backward compatibility with the previous API.
        # Existing code can still use model_dir=...
        if model_name_or_path is None:
            model_name_or_path = model_dir

        if model_name_or_path is None:
            raise ValueError(
                "Either model_name_or_path or model_dir must be provided."
            )

        self.model_name_or_path = str(model_name_or_path)
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.n_best = n_best
        self.max_answer_length = max_answer_length

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_path = Path(self.model_name_or_path)

        # If the path exists locally, load from the local directory.
        # Otherwise, treat it as a Hugging Face model ID.
        if model_path.exists():
            tokenizer_source = model_path
            model_source = model_path
        else:
            tokenizer_source = self.model_name_or_path
            model_source = self.model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=local_files_only,
        )

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )

        self.model.to(self.device)
        self.model.eval()

    def _build_features(
        self,
        question: str,
        context: str,
    ) -> List[Dict]:
        """
        Build explicit sliding-window features over the context.

        Sequence format:
            [CLS] question [SEP] context [SEP]

        The question and context are tokenized separately so long
        contexts can be handled explicitly without relying on
        tokenizer pair-overflow behavior.
        """

        question_encoding = self.tokenizer(
            question.strip(),
            add_special_tokens=False,
            truncation=False,
        )

        original_model_max_length = self.tokenizer.model_max_length

        self.tokenizer.model_max_length = int(1e9)

        try:
            context_encoding = self.tokenizer(
                context,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
        finally:
            self.tokenizer.model_max_length = original_model_max_length

        question_ids = question_encoding["input_ids"]
        context_ids = context_encoding["input_ids"]
        context_offsets = context_encoding["offset_mapping"]

        special_tokens_count = (
            self.tokenizer.num_special_tokens_to_add(pair=True)
        )

        max_context_tokens = (
            self.max_length
            - len(question_ids)
            - special_tokens_count
        )

        if max_context_tokens <= 0:
            raise ValueError(
                "The question is too long for the configured max_length."
            )

        step = max_context_tokens - self.doc_stride

        if step <= 0:
            raise ValueError(
                "doc_stride must be smaller than the available "
                "context window."
            )

        features = []
        start = 0
        feature_index = 0

        while start < len(context_ids):
            end = min(
                start + max_context_tokens,
                len(context_ids),
            )

            window_ids = context_ids[start:end]
            window_offsets = context_offsets[start:end]

            input_ids = (
                [self.tokenizer.cls_token_id]
                + question_ids
                + [self.tokenizer.sep_token_id]
                + window_ids
                + [self.tokenizer.sep_token_id]
            )

            attention_mask = [1] * len(input_ids)

            features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "context_offsets": window_offsets,
                    "context_token_start": start,
                    "context_token_end": end,
                    "feature_index": feature_index,
                }
            )

            feature_index += 1

            if end >= len(context_ids):
                break

            start += step

        return features

    def _generate_candidates(
        self,
        feature: Dict,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
    ) -> List[Dict]:
        """
        Generate valid start/end answer span candidates.

        Only tokens belonging to the context portion of the sequence
        are considered valid answer positions.
        """

        input_ids = feature["input_ids"]
        context_offsets = feature["context_offsets"]

        sep_token_id = self.tokenizer.sep_token_id

        separator_positions = [
            index
            for index, token_id in enumerate(input_ids)
            if token_id == sep_token_id
        ]

        if len(separator_positions) < 2:
            return []

        context_start_position = separator_positions[0] + 1
        context_end_position = separator_positions[1] - 1

        if context_end_position < context_start_position:
            return []

        valid_context_positions = list(
            range(
                context_start_position,
                context_end_position + 1,
            )
        )

        if not valid_context_positions:
            return []

        start_values = start_logits[valid_context_positions]
        end_values = end_logits[valid_context_positions]

        top_k = min(
            self.n_best,
            len(valid_context_positions),
        )

        top_start_indices = torch.topk(
            start_values,
            k=top_k,
        ).indices.tolist()

        top_end_indices = torch.topk(
            end_values,
            k=top_k,
        ).indices.tolist()

        candidates = []

        for start_offset in top_start_indices:
            start_position = valid_context_positions[start_offset]

            for end_offset in top_end_indices:
                end_position = valid_context_positions[end_offset]

                if end_position < start_position:
                    continue

                span_length = (
                    end_position - start_position + 1
                )

                if span_length > self.max_answer_length:
                    continue

                context_relative_start = (
                    start_position - context_start_position
                )

                context_relative_end = (
                    end_position - context_start_position
                )

                if context_relative_start < 0:
                    continue

                if context_relative_end >= len(context_offsets):
                    continue

                char_start = context_offsets[
                    context_relative_start
                ][0]

                char_end = context_offsets[
                    context_relative_end
                ][1]

                if char_end <= char_start:
                    continue

                score = (
                    start_logits[start_position].item()
                    + end_logits[end_position].item()
                )

                candidates.append(
                    {
                        "score": score,
                        "start_char": char_start,
                        "end_char": char_end,
                        "feature_index": feature[
                            "feature_index"
                        ],
                    }
                )

        return candidates

    def answer_question(
        self,
        question: str,
        context: str,
    ) -> Dict:
        """
        Answer a question using extractive QA.

        Returns:
            Dictionary containing:
            - answer
            - raw span score
            - character span
            - feature index
            - number of features
            - number of candidates
            - inference latency
        """

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if not context.strip():
            raise ValueError("Context cannot be empty.")

        start_time = time.perf_counter()

        features = self._build_features(
            question=question,
            context=context,
        )

        batch_input_ids = [
            feature["input_ids"]
            for feature in features
        ]

        batch_attention_masks = [
            feature["attention_mask"]
            for feature in features
        ]

        max_batch_length = max(
            len(input_ids)
            for input_ids in batch_input_ids
        )

        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            raise ValueError(
                "Tokenizer does not define a pad_token_id."
            )

        padded_input_ids = []
        padded_attention_masks = []

        for input_ids, attention_mask in zip(
            batch_input_ids,
            batch_attention_masks,
        ):
            padding_length = (
                max_batch_length - len(input_ids)
            )

            padded_input_ids.append(
                input_ids
                + [pad_token_id] * padding_length
            )

            padded_attention_masks.append(
                attention_mask
                + [0] * padding_length
            )

        input_tensor = torch.tensor(
            padded_input_ids,
            dtype=torch.long,
            device=self.device,
        )

        attention_tensor = torch.tensor(
            padded_attention_masks,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
            )

        all_candidates = []

        for feature_index, feature in enumerate(features):
            candidates = self._generate_candidates(
                feature=feature,
                start_logits=outputs.start_logits[
                    feature_index
                ],
                end_logits=outputs.end_logits[
                    feature_index
                ],
            )

            all_candidates.extend(candidates)

        inference_time = time.perf_counter() - start_time

        if not all_candidates:
            return {
                "answer": "",
                "score": None,
                "start_char": None,
                "end_char": None,
                "feature_index": None,
                "num_features": len(features),
                "num_candidates": 0,
                "inference_time": inference_time,
            }

        best_candidate = max(
            all_candidates,
            key=lambda candidate: candidate["score"],
        )

        answer = context[
            best_candidate["start_char"]:
            best_candidate["end_char"]
        ].strip()

        return {
            "answer": answer,
            "score": best_candidate["score"],
            "start_char": best_candidate["start_char"],
            "end_char": best_candidate["end_char"],
            "feature_index": best_candidate["feature_index"],
            "num_features": len(features),
            "num_candidates": len(all_candidates),
            "inference_time": inference_time,
        }

    def get_model_info(self) -> Dict:
        """Return basic model information for the application."""

        return {
            "model_name_or_path": self.model_name_or_path,
            "model_class": self.model.__class__.__name__,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "parameters": sum(
                parameter.numel()
                for parameter in self.model.parameters()
            ),
            "device": str(self.device),
            "max_length": self.max_length,
            "doc_stride": self.doc_stride,
            "n_best": self.n_best,
            "max_answer_length": self.max_answer_length,
        }