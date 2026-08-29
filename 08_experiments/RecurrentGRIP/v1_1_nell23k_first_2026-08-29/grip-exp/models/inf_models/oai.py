"""
Adapted from Microsoft GraphRAG.
https://github.com/microsoft/graphrag

"""
import asyncio
import json
from abc import abstractmethod
from datetime import datetime
from typing import Optional, Union, Any, cast

import openai
import tiktoken
from openai import AsyncOpenAI, OpenAI
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import trange

from .base import BaseInferenceModel

_MODEL_REQUIRED_MSG = "model is required"

OPENAI_RETRY_ERROR_TYPES = (
    cast(Any, openai).RateLimitError,
    cast(Any, openai).APIConnectionError,
)


class BaseOpenAILLM(BaseInferenceModel):
    """The Base OpenAI LLM implementation."""

    _async_client: AsyncOpenAI
    _sync_client: OpenAI

    def __init__(self):
        super().__init__()
        self._create_openai_client()

    @abstractmethod
    def _create_openai_client(self):
        """Create a new synchronous and asynchronous OpenAI client instance."""

    def set_clients(
            self,
            sync_client: OpenAI,
            async_client: AsyncOpenAI,
    ):
        """
        Set the synchronous and asynchronous clients used for making API requests.

        Args:
            sync_client (OpenAI | AzureOpenAI): The sync client object.
            async_client (AsyncOpenAI | AsyncAzureOpenAI): The async client object.
        """
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def async_client(self) -> Optional[AsyncOpenAI]:
        """
        Get the asynchronous client used for making API requests.

        Returns
        -------
            AsyncOpenAI | AsyncAzureOpenAI: The async client object.
        """
        return self._async_client

    @property
    def sync_client(self) -> Optional[OpenAI]:
        """
        Get the synchronous client used for making API requests.

        Returns
        -------
            AsyncOpenAI | AsyncAzureOpenAI: The async client object.
        """
        return self._sync_client

    @async_client.setter
    def async_client(self, client: AsyncOpenAI):
        """
        Set the asynchronous client used for making API requests.

        Args:
            client (AsyncOpenAI | AsyncAzureOpenAI): The async client object.
        """
        self._async_client = client

    @sync_client.setter
    def sync_client(self, client: OpenAI):
        """
        Set the synchronous client used for making API requests.

        Args:
            client (OpenAI | AzureOpenAI): The sync client object.
        """
        self._sync_client = client

    @abstractmethod
    def inference(
            self,
            user_contents: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None):
        return


class ChatOpenAI(BaseOpenAILLM):
    def __init__(
            self,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            api_version: Optional[str] = None,
            organization: Optional[str] = None,
            model: Optional[str] = None,
            gen_max_length: int = 10000,
            batch_size: int = -1,
            max_retries: int = 10,
            request_timeout: float = 180.0,
            retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
            **kwargs,
    ):

        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.organization = organization
        self.model = model
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.retry_error_types = retry_error_types
        self.gen_max_length = gen_max_length
        self.batch_size = batch_size
        super().__init__()
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def _create_openai_client(self):
        sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            organization=self.organization,
            # Retry Configuration
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )

        async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            organization=self.organization,
            # Retry Configuration
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )
        self.set_clients(sync_client=sync_client, async_client=async_client)

    def create_single_input(
            self,
            user_content: str,
            system_prompt: Optional[str] = None,
            history: Optional[list] = None) -> list[dict]:
        if history is not None:
            message = history + [{"role": "user", "content": user_content}]
        else:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        return message

    def _generate(
            self,
            messages: Union[str, list[Any]],
            **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = self.sync_client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            stream=False,
            **kwargs,
        )  # type: ignore
        return response.choices[0].message.content or ""  # type: ignore

    async def _agenerate(
            self,
            messages: Union[str, list[Any]],
            **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = await self.async_client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""  # type: ignore

    def generate(
            self,
            messages: Union[str, list[Any]],
            **kwargs: Any,
    ) -> dict:
        """Generate text."""

        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    response = self._generate(
                        messages=messages,
                        **kwargs,
                    )
                    return {
                        "query": messages[1]["content"],
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }

        except RetryError as e:
            return {
                "query": messages[1]["content"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
        else:
            # TODO: why not just throw in this case?
            return ""

    async def agenerate(
            self,
            messages: Union[str, list[Any]],
            **kwargs: Any,
    ) -> dict:
        """Generate text asynchronously."""
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),  # type: ignore
            )
            async for attempt in retryer:
                with attempt:
                    response = await self._agenerate(
                        messages=messages,
                        **kwargs,
                    )
                    return {
                        "query": messages[1]["content"],
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }
        except RetryError as e:
            return {
                "query": messages[1]["content"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
        else:
            # TODO: why not just throw in this case?
            return ""

    def batch_inference(
            self,
            batch_input: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None
    ) -> list[dict]:
        if histories is not None:
            batch_input = [self.create_single_input(input_text, system_prompt, history)
                           for input_text, history in zip(batch_input, histories)]
        else:
            batch_input = [self.create_single_input(input_text, system_prompt) for input_text in batch_input]
        responses = [self.generate(message) for message in batch_input]
        return responses

    async def abatch_inference(
            self,
            batch_input: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None
    ) -> list[dict]:
        if histories is not None:
            batch_input = [self.create_single_input(input_text, system_prompt, history)
                           for input_text, history in zip(batch_input, histories)]
        else:
            batch_input = [self.create_single_input(input_text, system_prompt) for input_text in batch_input]
        responses = await asyncio.gather(*[self.agenerate(message) for message in batch_input])
        return responses

    def inference(
            self,
            user_contents: list[Union[str, dict]],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None
    ) -> list[dict]:
        if isinstance(user_contents[0], dict):
            user_contents = [user_content["query"] for user_content in user_contents]
        if self.batch_size == -1:
            return asyncio.run(self.abatch_inference(user_contents, system_prompt, histories))
        else:
            results = []
            for start_index in trange(0, len(user_contents), self.batch_size, desc=f"Batches", disable=False, ):
                batch_contents = user_contents[start_index: start_index + self.batch_size]
                batch_histories = histories[start_index: start_index + self.batch_size] if histories else None
                batch_result = asyncio.run(self.abatch_inference(batch_contents, system_prompt, batch_histories))
                results.extend(batch_result)
            return results

    def save_results(self, results: list[dict], filename: str):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
