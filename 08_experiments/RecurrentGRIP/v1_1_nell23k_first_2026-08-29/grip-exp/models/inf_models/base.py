from abc import abstractmethod, ABC
from typing import Optional


class BaseInferenceModel(ABC):
    """Base inference model class. All model must implement inference function.
    """

    @abstractmethod
    def inference(
            self,
            user_contents: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None):
        return
