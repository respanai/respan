"""Respan Gateway Generator for Haystack pipelines."""

from typing import Any, Dict, List, Optional

import requests
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage

from respan_sdk.constants import resolve_chat_completions_endpoint

from respan_exporter_haystack.utils.chat_utils import (
    extract_response_text,
    to_chat_message,
    to_request_message,
)
from respan_exporter_haystack.utils.config_utils import resolve_api_key, resolve_base_url

logger = logging.getLogger(__name__)


class _BaseRespanGenerator(object):
    """Base class for Respan generators containing shared logic."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_id: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = resolve_api_key(api_key=api_key)
        self.base_url = resolve_base_url(base_url=base_url)
        self.prompt_id = prompt_id
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError(
                "Respan API key is required. Set RESPAN_API_KEY environment variable "
                "or pass api_key parameter."
            )
            
        if not self.model and not self.prompt_id:
            raise ValueError(
                "Either 'model' or 'prompt_id' must be provided. "
                "Use 'model' for direct model calls, or 'prompt_id' to use platform-managed prompts."
            )
        
        self.endpoint = resolve_chat_completions_endpoint(base_url=self.base_url)

    def _build_payload(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the request payload for the Respan API."""
        kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        payload = {**kwargs}
        
        if messages is not None:
            payload["messages"] = messages
            
        if self.model:
            payload["model"] = self.model
            
        if self.prompt_id:
            prompt_config = {
                "prompt_id": self.prompt_id,
                "override": True,
            }
            if prompt_variables:
                prompt_config["variables"] = prompt_variables
            payload["prompt"] = prompt_config
            
        return {k: v for k, v in payload.items() if v is not None}

    def _execute_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the HTTP request to the Respan API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            logger.debug("Calling Respan gateway with model %s", self.model)
            response = requests.post(
                url=self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from Respan: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except requests.exceptions.Timeout:
            error_msg = "Request to Respan timed out"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error calling Respan gateway: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _extract_metadata(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract metadata from the API response."""
        meta = []
        usage = data.get("usage", {})
        choices = data.get("choices", [])
        
        for i, choice in enumerate(choices):
            meta.append({
                "model": data.get("model", self.model),
                "index": i,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cost": data.get("cost"),
            })
        return meta

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary. API key is never serialized to avoid leaking secrets."""
        return default_to_dict(
            obj=self,
            model=self.model,
            api_key=None,  # Never serialize; resolved from env on from_dict
            base_url=self.base_url,
            prompt_id=self.prompt_id,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
        )


@component
class RespanGenerator(_BaseRespanGenerator):
    """
    A Haystack Generator component that routes LLM calls through Respan gateway.
    
    This replaces OpenAIGenerator and routes all calls through Respan for:
    - Automatic logging
    - Fallbacks and retries
    - Load balancing
    - Cost optimization
    - Prompt management (use platform-managed prompts)
    - All Respan platform features
    
    Example usage:
        ```python
        from respan_exporter_haystack.gateway import RespanGenerator
        
        # Basic usage
        generator = RespanGenerator(
            model="gpt-4o-mini",
            api_key="your-respan-api-key"
        )
        result = generator.run(messages=[{"role": "user", "content": "Hello!"}])
        
        # With platform-managed prompts
        generator = RespanGenerator(
            model="gpt-4o-mini",
            prompt_id="042f5f",  # Prompt from Respan platform
            api_key="your-respan-api-key"
        )
        result = generator.run(prompt_variables={"customer_name": "John"})
        ```
    
    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-4"). Optional if using prompt_id.
        api_key: Respan API key (defaults to RESPAN_API_KEY env var)
        base_url: Respan API base URL (defaults to https://api.respan.ai)
        prompt_id: Optional prompt ID from Respan platform for prompt management
        generation_kwargs: Additional parameters (temperature, max_tokens, etc.)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_id: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ):
        """Initialize the Respan gateway generator."""
        # Explicit base call required: Haystack's @component decorator changes MRO, so super()
        # would not resolve to _BaseRespanGenerator. Do not replace with super().
        _BaseRespanGenerator.__init__(
            self,
            model=model,
            api_key=api_key,
            base_url=base_url,
            prompt_id=prompt_id,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
        )

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text using Respan gateway.
        
        Args:
            prompt: Simple prompt string (will be converted to user message)
            messages: List of message dicts with 'role' and 'content'
            generation_kwargs: Additional generation parameters (overrides init kwargs)
            prompt_variables: Variables for platform-managed prompt (requires prompt_id in init)
            
        Returns:
            Dictionary with:
                - replies: List of generated texts
                - meta: List of metadata dicts (model, tokens, cost, etc.)
        """
        # Set up default messages
        _messages = messages
        if _messages is None:
            _messages = []
        
        # Handle prompt management mode
        if not self.prompt_id:
            # Convert prompt to messages format if provided
            if prompt is not None:
                _messages = [{"role": "user", "content": prompt}]
            elif not _messages:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
        else:
            if prompt is not None:
                logger.warning("Prompt ID provided alongside 'prompt' arg. The explicit 'prompt' will be ignored.")
            if messages:
                logger.warning("Prompt ID provided alongside 'messages' arg. Explicit messages will be appended/ignored according to platform rules.")
                _messages = messages
            else:
                _messages = None
        
        payload = self._build_payload(
            messages=_messages,
            generation_kwargs=generation_kwargs,
            prompt_variables=prompt_variables,
        )
        
        data = self._execute_request(payload)
        
        choices = data.get("choices", [])
        replies = [
            extract_response_text(content=choice.get("message", {}).get("content"))
            for choice in choices
        ]
        
        logger.debug("Successfully generated %s replies", len(replies))
        
        return {
            "replies": replies,
            "meta": self._extract_metadata(data),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RespanGenerator":
        """Deserialize component from dictionary."""
        return default_from_dict(cls=cls, data=data)


@component
class RespanChatGenerator(_BaseRespanGenerator):
    """
    Respan Chat Generator for Haystack pipelines.
    
    Similar to RespanGenerator but with chat-specific features.
    Use this when you want ChatMessage support and chat-specific parameters.
    
    Example:
        ```python
        from haystack.dataclasses import ChatMessage
        from respan_exporter_haystack.gateway import RespanChatGenerator
        
        generator = RespanChatGenerator(
            model="gpt-4",
            api_key="your-respan-api-key"
        )
        
        messages = [
            ChatMessage.from_system("You are helpful"),
            ChatMessage.from_user("Hello!")
        ]
        
        result = generator.run(messages=messages)
        ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompt_id: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ):
        """Initialize the chat generator."""
        # Explicit base call required: Haystack's @component decorator changes MRO, so super()
        # would not resolve to _BaseRespanGenerator. Do not replace with super().
        _BaseRespanGenerator.__init__(
            self,
            model=model,
            api_key=api_key,
            base_url=base_url,
            prompt_id=prompt_id,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
        )

    @component.output_types(replies=List[ChatMessage], meta=List[Dict[str, Any]])
    def run(
        self,
        messages: Optional[List[ChatMessage]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate chat responses using Respan gateway.
        
        Args:
            messages: List of ChatMessage objects (optional if using prompt_id)
            generation_kwargs: Additional generation parameters
            prompt_variables: Variables for platform-managed prompt (requires prompt_id in init)
            
        Returns:
            Dictionary with:
                - replies: List of ChatMessage objects
                - meta: List of metadata dicts
        """
        # Handle prompt management mode
        messages_dict = None
        if not self.prompt_id:
            if not messages:
                raise ValueError("Either 'messages' must be provided or 'prompt_id' must be set during initialization")
            # Convert ChatMessage to dict format
            messages_dict = [
                to_request_message(message=msg)
                for msg in messages
            ]
        else:
            if messages:
                logger.warning("Prompt ID provided alongside 'messages' arg. Explicit messages will be appended/ignored according to platform rules.")
                messages_dict = [
                    to_request_message(message=msg)
                    for msg in messages
                ]
            else:
                messages_dict = None
        
        payload = self._build_payload(
            messages=messages_dict,
            generation_kwargs=generation_kwargs,
            prompt_variables=prompt_variables,
        )
        
        data = self._execute_request(payload)
        
        choices = data.get("choices", [])
        replies = []
        
        for choice in choices:
            msg_data = choice.get("message", {})
            replies.append(to_chat_message(message_payload=msg_data))
        
        logger.debug("Successfully generated %s replies", len(replies))
        
        return {
            "replies": replies,
            "meta": self._extract_metadata(data),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RespanChatGenerator":
        """Deserialize component from dictionary."""
        return default_from_dict(cls=cls, data=data)
