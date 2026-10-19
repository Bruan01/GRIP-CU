import numpy as np
import torch
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

from constants import HF_EMBEDDING_LLMS
from .base import BaseEmbeddingModel


class HFEmbeddingModel(BaseEmbeddingModel):
    """Embedding model using huggingface framework.
    Args:
        model_name: Name of the model

    """
    SUPPORTED_MODELS = HF_EMBEDDING_LLMS

    def __init__(
            self,
            model_name: str = "bge_large",
            batch_size: int = -1,
            tokenize_max_length: int = 2048,
            **kwargs
    ):
        super().__init__()
        if model_name in self.SUPPORTED_MODELS:
            self.model_name = self.SUPPORTED_MODELS[model_name]
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.batch_size = batch_size
        self.tokenize_max_length = tokenize_max_length

    def batch_encode(self, batch_texts: list[str]) -> np.ndarray:
        encoded_input = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.tokenize_max_length)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy()

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.batch_size == -1:
            return self.batch_encode(texts)
        else:
            results = []
            for start_index in trange(0, len(texts), self.batch_size, desc=f"Batch", disable=False, ):
                batch_texts = texts[start_index: start_index + self.batch_size]
                batch_result = self.batch_encode(batch_texts)
                results.append(batch_result)

            return np.concatenate(results, axis=0)
