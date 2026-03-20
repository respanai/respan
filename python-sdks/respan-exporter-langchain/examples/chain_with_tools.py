"""
Chain with Tools: Trace a LangChain agent that uses tool calling with Respan.

This example creates an agent with custom tools and traces the full
workflow — including tool invocations — to Respan.

Prerequisites:
    pip install respan-exporter-langchain langchain-openai

Environment variables:
    RESPAN_API_KEY   - Your Respan API key
    OPENAI_API_KEY   - Your OpenAI API key
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from respan_exporter_langchain import RespanCallbackHandler

# 1. Define custom tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is 72°F and sunny."

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {"new york": "8.3 million", "london": "8.8 million"}
    return populations.get(city.lower(), "Unknown")

# 2. Create the Respan callback handler
handler = RespanCallbackHandler(
    api_key="your-respan-api-key",          # or set RESPAN_API_KEY env var
    trace_name="tool-calling-agent",
)

# 3. Build a tool-calling agent
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[get_weather, get_population])

# 4. Invoke the agent with the Respan callback
result = agent.invoke(
    {"messages": [("human", "What is the weather and population of New York?")]},
    config={"callbacks": [handler]},
)

for message in result["messages"]:
    print(f"{message.type}: {message.content}")
