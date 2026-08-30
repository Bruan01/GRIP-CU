from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TaskDataset(Dataset):
    """Training task dataset for LLM fine-tuning.

    """

    def __init__(self,
                 context_samples: list[str],
                 qa_samples: list[str],
                 tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.context_samples = context_samples
        self.qa_samples = qa_samples
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
            self.data_list = self.qa_samples  # + self.context_samples
        else:
            self.data_list = self.context_samples

    def tokenize_function(self, text):
        return self.tokenizer(text, truncation=True, padding=False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.tokenize_function(self.data_list[idx])
