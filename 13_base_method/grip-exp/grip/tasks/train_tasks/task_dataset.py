from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TaskDataset(Dataset):
    """Training task dataset for LLM fine-tuning.

    """

    def __init__(self,
                 context_samples: list[str],
                 qa_samples: list[str],
                 tokenizer: PreTrainedTokenizer,
                 candidate_samples: list[dict | None] | None = None):
        super().__init__()
        self.context_samples = context_samples
        self.qa_samples = qa_samples
        self.candidate_samples = candidate_samples or []
        if self.candidate_samples and len(self.candidate_samples) != len(self.qa_samples):
            raise ValueError(
                'Candidate records must be aligned one-to-one with QA samples: '
                f'{len(self.candidate_samples)} != {len(self.qa_samples)}.'
            )
        self.data_list = context_samples
        self.tokenizer = tokenizer
        self._enable_qa = False

    @property
    def enable_qa(self):
        return self._enable_qa

    @enable_qa.setter
    def enable_qa(self, value: bool):
        self._enable_qa = value
        if value:
            self.data_list = self.qa_samples
        else:
            self.data_list = self.context_samples

    def tokenize_function(self, text):
        return self.tokenizer(text, truncation=True, padding=False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.tokenize_function(self.data_list[idx])
        if self._enable_qa and self.candidate_samples:
            item["candidate_record"] = self.candidate_samples[idx]
        return item
