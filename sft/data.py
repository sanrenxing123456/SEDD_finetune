"""Data preparation helpers for block-wise conditional SEDD fine-tuning."""

import json
import os
import random

import torch
from torch.utils.data import DataLoader


QUESTION_PREFIX = "Question:\n"
REASONING_PREFIX = "\n\nReasoning:\n"
FINAL_ANSWER_PREFIX = "\n\nFinal answer:\n"


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value if item is not None).strip()
    return str(value).strip()


def first_text(example, field_names):
    """Return the first non-empty text field from a dataset row."""
    for name in field_names:
        text = _as_text(example.get(name))
        if text:
            return text
    return ""


def extract_s1_text(example, thinking_fields=None, answer_fields=None):
    """Extract a question and a teacher response from s1K or s1K-1.1."""
    thinking_fields = thinking_fields or (
        "deepseek_thinking_trajectory",
        "thinking_trajectories",
        "gemini_thinking_trajectory",
    )
    answer_fields = answer_fields or (
        "deepseek_attempt",
        "attempt",
        "gemini_attempt",
        "solution",
    )
    question = _as_text(example.get("question"))
    reasoning = first_text(example, thinking_fields)
    answer = first_text(example, answer_fields)
    if not question or not (reasoning or answer):
        return None
    if reasoning and answer:
        response = reasoning + FINAL_ANSWER_PREFIX + answer
    else:
        response = reasoning or answer
    return question, response


def encode_prompt(tokenizer, question, max_question_tokens):
    prefix = tokenizer.encode(QUESTION_PREFIX, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    suffix = tokenizer.encode(REASONING_PREFIX, add_special_tokens=False)
    if max_question_tokens > 0:
        question_ids = question_ids[:max_question_tokens]
    return prefix, question_ids, suffix


def make_context_ids(
    tokenizer,
    question,
    history_ids,
    context_length,
    max_question_tokens,
):
    """Fit the question plus the most recent reasoning history into a context."""
    prefix, question_ids, suffix = encode_prompt(
        tokenizer, question, max_question_tokens
    )
    fixed_size = len(prefix) + len(suffix)
    question_budget = max(context_length - fixed_size, 0)
    question_ids = question_ids[:question_budget]
    history_budget = max(context_length - fixed_size - len(question_ids), 0)
    history_ids = list(history_ids[-history_budget:]) if history_budget else []
    context_ids = prefix + question_ids + suffix + history_ids
    return context_ids[-context_length:]


def build_chunk_record(
    tokenizer,
    question,
    response_ids,
    chunk_index,
    max_length=1024,
    target_length=256,
    max_question_tokens=384,
    question_id=-1,
):
    """Build one fixed-length conditional-denoising training record."""
    if target_length <= 0 or target_length >= max_length:
        raise ValueError("target_length must be between 1 and max_length - 1")
    start = chunk_index * target_length
    target_ids = list(response_ids[start : start + target_length])
    if not target_ids:
        raise ValueError("chunk_index points beyond the response")

    is_final = start + target_length >= len(response_ids)
    if is_final and (not target_ids or target_ids[-1] != tokenizer.eos_token_id):
        if len(target_ids) == target_length:
            raise ValueError("response_ids must reserve room for EOS in its final chunk")
        target_ids.append(tokenizer.eos_token_id)

    context_length = max_length - target_length
    context_ids = make_context_ids(
        tokenizer,
        question,
        response_ids[:start],
        context_length,
        max_question_tokens,
    )
    target_start = len(context_ids)
    input_ids = context_ids + target_ids
    condition_mask = [True] * len(context_ids) + [False] * len(target_ids)
    loss_mask = [False] * len(context_ids) + [True] * len(target_ids)

    padding = max_length - len(input_ids)
    input_ids += [tokenizer.eos_token_id] * padding
    condition_mask += [True] * padding
    loss_mask += [False] * padding

    return {
        "input_ids": input_ids,
        "condition_mask": condition_mask,
        "loss_mask": loss_mask,
        "question_id": int(question_id),
        "chunk_index": int(chunk_index),
        "target_start": int(target_start),
        "is_final": bool(is_final),
    }


def build_inference_record(
    tokenizer,
    question,
    history_ids,
    max_length=1024,
    target_length=256,
    max_question_tokens=384,
):
    """Build a conditioning sequence whose target block will start as MASK."""
    context_length = max_length - target_length
    context_ids = make_context_ids(
        tokenizer,
        question,
        history_ids,
        context_length,
        max_question_tokens,
    )
    target_start = len(context_ids)
    input_ids = context_ids + [tokenizer.eos_token_id] * target_length
    condition_mask = [True] * len(context_ids) + [False] * target_length
    padding = max_length - len(input_ids)
    input_ids += [tokenizer.eos_token_id] * padding
    condition_mask += [True] * padding
    return {
        "input_ids": input_ids,
        "condition_mask": condition_mask,
        "target_start": target_start,
        "target_length": target_length,
    }


def expand_split(
    rows,
    tokenizer,
    max_length,
    target_length,
    max_question_tokens,
    max_chunks_per_example=0,
):
    records = []
    skipped = 0
    for question_id, row in rows:
        extracted = extract_s1_text(row)
        if extracted is None:
            skipped += 1
            continue
        question, response = extracted
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if not response_ids:
            skipped += 1
            continue
        if response_ids[-1] != tokenizer.eos_token_id:
            response_ids.append(tokenizer.eos_token_id)
        num_chunks = (len(response_ids) + target_length - 1) // target_length
        if max_chunks_per_example > 0:
            num_chunks = min(num_chunks, max_chunks_per_example)
        for chunk_index in range(num_chunks):
            records.append(
                build_chunk_record(
                    tokenizer=tokenizer,
                    question=question,
                    response_ids=response_ids,
                    chunk_index=chunk_index,
                    max_length=max_length,
                    target_length=target_length,
                    max_question_tokens=max_question_tokens,
                    question_id=question_id,
                )
            )
    return records, skipped


def split_rows(dataset, seed=42, train_ratio=0.8, validation_ratio=0.1):
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("split ratios must leave a non-empty test fraction")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    train_end = int(len(indices) * train_ratio)
    validation_end = train_end + int(len(indices) * validation_ratio)
    groups = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }
    return {
        name: [(index, dataset[index]) for index in split_indices]
        for name, split_indices in groups.items()
    }


def prepare_s1_dataset(
    dataset_name,
    output_dir,
    tokenizer_name="gpt2",
    cache_dir=None,
    seed=42,
    train_ratio=0.8,
    validation_ratio=0.1,
    max_length=1024,
    target_length=256,
    max_question_tokens=384,
    max_chunks_per_example=0,
):
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    raw = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    row_splits = split_rows(raw, seed, train_ratio, validation_ratio)

    processed = {}
    stats = {}
    for split_name, rows in row_splits.items():
        records, skipped = expand_split(
            rows,
            tokenizer,
            max_length,
            target_length,
            max_question_tokens,
            max_chunks_per_example,
        )
        if not records:
            raise ValueError(f"No usable records were produced for split {split_name}")
        processed[split_name] = Dataset.from_list(records)
        stats[split_name] = {
            "questions": len(rows),
            "chunks": len(records),
            "skipped_questions": skipped,
            "final_chunks": sum(record["is_final"] for record in records),
        }

    os.makedirs(output_dir, exist_ok=True)
    DatasetDict(processed).save_to_disk(output_dir)
    metadata = {
        "dataset_name": dataset_name,
        "tokenizer_name": tokenizer_name,
        "seed": seed,
        "max_length": max_length,
        "target_length": target_length,
        "max_question_tokens": max_question_tokens,
        "max_chunks_per_example": max_chunks_per_example,
        "stats": stats,
    }
    with open(os.path.join(output_dir, "sft_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def load_sft_metadata(data_dir):
    with open(os.path.join(data_dir, "sft_metadata.json"), encoding="utf-8") as handle:
        return json.load(handle)


def make_dataloader(dataset, batch_size, shuffle, num_workers=0):
    columns = ["input_ids", "condition_mask", "loss_mask"]
    dataset = dataset.with_format("torch", columns=columns, output_all_columns=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
