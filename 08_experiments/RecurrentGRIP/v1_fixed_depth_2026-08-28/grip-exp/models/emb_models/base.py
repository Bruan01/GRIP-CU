from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Base class for embedding model. All model should implement encode function.
    """

    @abstractmethod
    def encode(self, texts: list[str]):
        raise NotImplementedError("Please implement encode method!")
