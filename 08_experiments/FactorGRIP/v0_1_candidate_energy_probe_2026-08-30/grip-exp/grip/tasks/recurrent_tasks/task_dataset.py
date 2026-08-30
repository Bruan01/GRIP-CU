from __future__ import annotations

from constants import ANSWER_TAG, SYSTEM_PROMPT
from grip.tasks.train_tasks.task_dataset import TaskDataset


QUESTION_TEMPLATE = (
    "Given the context graph titled {title}, please answer the following question: "
    "{question} Response in the following format:<" + ANSWER_TAG + ">[answer]</" + ANSWER_TAG + ">"
)


def format_recurrent_qa(tokenizer, title: str, question: str, answer: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION_TEMPLATE.format(title=title, question=question)},
        {"role": "assistant", "content": f"<{ANSWER_TAG}>{answer}</{ANSWER_TAG}>"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_recurrent_task_dataset(tokenizer, title: str, context_samples: list[str], samples: list[dict]):
    qa_samples = [
        format_recurrent_qa(tokenizer, title, sample["question"], sample["answer"])
        for sample in samples
        if sample["split"] == "train"
    ]
    if not qa_samples:
        raise ValueError("recurrent training split is empty")
    eos = tokenizer.eos_token or ""
    qa_samples = [sample if not eos or sample.endswith(eos) else sample + eos for sample in qa_samples]
    return TaskDataset(context_samples=context_samples, qa_samples=qa_samples, tokenizer=tokenizer)
