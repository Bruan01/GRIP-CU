"""
LLM inference with Claude API.
"""

import asyncio
import json
from abc import abstractmethod
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, cast

import anthropic
from anthropic import AsyncAnthropic, Anthropic
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
CLAUDE_RETRY_ERROR_TYPES = (
    cast(Any, anthropic).RateLimitError,
    cast(Any, anthropic).APIConnectionError,
)


class BaseClaudeLLM(BaseInferenceModel):
    """The Base Claude LLM implementation."""

    _async_client: AsyncAnthropic
    _sync_client: Anthropic

    def __init__(self):
        super().__init__()
        self._create_claude_client()

    @abstractmethod
    def _create_claude_client(self):
        """Create a new synchronous and asynchronous Claude client instance."""

    def set_clients(
            self,
            sync_client: Anthropic,
            async_client: AsyncAnthropic,
    ):
        """
        Set the synchronous and asynchronous clients used for making API requests.

        Args:
            sync_client: The sync client object.
            async_client: The async client object.
        """
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def async_client(self) -> Optional[AsyncAnthropic]:
        """
        Get the asynchronous client used for making API requests.
        """
        return self._async_client

    @property
    def sync_client(self) -> Optional[Anthropic]:
        """
        Get the synchronous client used for making API requests.
        """
        return self._sync_client

    @async_client.setter
    def async_client(self, client: AsyncAnthropic):
        """
        Set the asynchronous client used for making API requests.
        Args:
            client: The async client object.
        """
        self._async_client = client

    @sync_client.setter
    def sync_client(self, client: Anthropic):
        """
        Set the synchronous client used for making API requests.
        Args:
            client: The sync client object.
        """
        self._sync_client = client

    @abstractmethod
    def inference(
            self,
            user_contents: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None):
        return


class ChatClaude(BaseClaudeLLM):
    """Chat Claude LLM.
    Args:
        api_key: API key.
        model: The name of the Claude model.
        batch_size: Inference batch size.
        gen_max_length: Maximum response length.
        max_retries: Maximum API call retry before throughout error.
        request_timeout: API call listen time before failed in sceond.
        retry_error_types: error type for retry.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            max_retries: int = 10,
            request_timeout: float = 180.0,
            gen_max_length: int = 10000,
            batch_size: int = -1,
            retry_error_types: tuple[type[BaseException]] = CLAUDE_RETRY_ERROR_TYPES,  # type: ignore
            **kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.retry_error_types = retry_error_types
        self.gen_max_length = gen_max_length
        self.batch_size = batch_size
        super().__init__()

    def _create_claude_client(self):
        sync_client = anthropic.Anthropic(api_key=self.api_key)
        async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.set_clients(sync_client=sync_client, async_client=async_client)

    def create_single_input(
            self,
            user_content: str,
            system_prompt: Optional[str] = None,
            history: Optional[list] = None
    ) -> tuple[str, list[dict]]:
        """Create chat input using template for single input.
        Args:
            user_content: user content.
            system_prompt: system prompt.
            history: Conversation history.
        """
        if history is not None:
            history = history + [{"role": "user", "content": user_content}]
        else:
            history = [{"role": "user", "content": user_content}]
        message = (
            system_prompt,
            history
        )
        return message

    def _generate(
            self,
            messages: Union[str, list[Any]],
            **kwargs,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = self.sync_client.messages.create(
            model=model,
            max_tokens=self.gen_max_length,
            system=messages[0],
            messages=messages[1],
        )
        return response.content[0].text or ""  # type: ignore

    async def _agenerate(
            self,
            messages: Union[str, list[Any]],
            **kwargs,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = await self.async_client.messages.create(  # type: ignore
            model=model,
            max_tokens=self.gen_max_length,
            system=messages[0],
            messages=messages[1],  # type: ignore
            **kwargs,
        )
        return response.content[0].text or ""  # type: ignore

    def generate(
            self,
            messages: Union[str, list[Any]],
            **kwargs,
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
                        "query": messages[1][0]["content"],
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }

        except RetryError as e:
            return {
                "query": messages[1][0]["content"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
        else:
            return ""

    async def agenerate(
            self,
            messages: Union[str, list[Any]],
            **kwargs,
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
                        "query": messages[1][0]["content"],
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success"
                    }
        except RetryError as e:
            return {
                "query": messages[1][0]["content"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
        else:
            return ""

    def batch_inference(
            self,
            batch_input: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None,
    ) -> list[dict]:
        batch_input = [self.create_single_input(input_text, system_prompt, history)
                       for input_text, history in zip(batch_input, histories)]
        responses = [self.generate(message) for message in batch_input]
        return responses

    async def abatch_inference(
            self,
            batch_input: list[str],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None,
    ) -> list[dict]:
        batch_input = [self.create_single_input(input_text, system_prompt, history)
                       for input_text, history in zip(batch_input, histories)]
        responses = await asyncio.gather(*[self.agenerate(message) for message in batch_input])
        return responses

    def inference(
            self,
            user_contents: list[Union[str, dict]],
            system_prompt: Optional[str] = None,
            histories: Optional[list[list]] = None,
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

    def save_results(self, results: List[Dict], filename: str):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
