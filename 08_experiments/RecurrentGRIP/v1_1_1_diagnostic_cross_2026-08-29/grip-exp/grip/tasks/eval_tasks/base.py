from typing import Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from constants import SYSTEM_PROMPT, ANSWER_TAG


class BaseEvalDataset(Dataset):
    system_prompt = SYSTEM_PROMPT

    answer_format = f"<{ANSWER_TAG}>[answer]</{ANSWER_TAG}>"

    def __init__(
            self,
            questions: list,
            answers: list,
            tokenizer: PreTrainedTokenizer,
            graph: dict,
            title: Optional[str] = None,
            **kwargs,
    ):
        super().__init__()
        assert len(questions) == len(
            answers), f"questions and answers must have the same length, but got {len(questions)} and {len(answers)}"
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.graph = graph
        self.title = title

    def __len__(self):
        return len(self.questions)

    def _format_chat_template(self, user_content):
        message = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_content
            }]

        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, )
