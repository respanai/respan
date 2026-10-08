"""Real AgentSpec/LangGraph runtime with a deterministic LangChain model."""

from contextlib import contextmanager
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pyagentspec.adapters.langgraph import AgentSpecLoader
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig
from pyagentspec.property import FloatProperty
from pyagentspec.tools import ServerTool
from pydantic import Field


class FixtureModel(BaseChatModel):
    bound_tools: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "fixture"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(
            update={"bound_tools": [convert_to_openai_tool(tool) for tool in tools]}
        )

    def _get_invocation_params(self, **kwargs):
        parameters = super()._get_invocation_params(**kwargs)
        if self.bound_tools:
            parameters["tools"] = self.bound_tools
        return parameters

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        question = str(messages[-1].content)
        if "model-failure" in question:
            raise ValueError("controlled model failure")
        if "tool-call" in question and messages[-1].type != "tool":
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "fixture-call-1",
                        "name": "subtract",
                        "args": {"left": 3.0, "right": 3.0},
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="answer " + question)
        message.usage_metadata = {
            "input_tokens": 0,
            "output_tokens": 3,
            "total_tokens": 3,
            "input_token_details": {"cache_read": 0},
            "output_token_details": {"reasoning": 2},
        }
        if message.tool_calls:
            message.additional_kwargs["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": __import__("json").dumps(call["args"]),
                    },
                }
                for call in message.tool_calls
            ]
        message.id = "fixture-response"
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=message,
                    generation_info={
                        "finish_reason": "tool_calls" if message.tool_calls else "stop"
                    },
                )
            ]
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for text in ["fixture ", "stream"]:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=text, id="fixture-stream")
            )
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id="fixture-stream",
                usage_metadata={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            ),
            generation_info={"finish_reason": "stop"},
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk


@contextmanager
def fixture_model():
    import pyagentspec.adapters.langgraph._langgraphconverter as converter

    def factory(**kwargs):
        return FixtureModel(callbacks=kwargs.get("callbacks"))

    if hasattr(converter, "_create_chat_openai_model"):
        with patch.object(converter, "_create_chat_openai_model", factory):
            yield
    else:
        with patch("langchain_openai.ChatOpenAI", factory):
            yield


def build_agent(name="fixture-agent", *, with_tool=False, tool_fails=False):
    tool = ServerTool(
        name="subtract",
        description="Subtract fixture numbers.",
        inputs=[FloatProperty(title="left"), FloatProperty(title="right")],
        outputs=[FloatProperty(title="difference")],
    )
    config = OpenAiConfig(
        name="fixture-config",
        model_id="fixture-model",
        api_key="never-export-config-secret",
    )
    agent = Agent(
        name=name,
        llm_config=config,
        system_prompt="fixture system",
        tools=[tool] if with_tool else [],
    )

    def subtract(left, right):
        if tool_fails:
            raise ValueError("controlled tool failure")
        return left - right

    return AgentSpecLoader(tool_registry={"subtract": subtract}).load_component(agent)
