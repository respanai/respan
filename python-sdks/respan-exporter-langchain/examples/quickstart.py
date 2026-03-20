"""
Quickstart: Trace a LangChain LLM call with Respan.

Prerequisites:
    pip install respan-exporter-langchain langchain-openai

Environment variables:
    RESPAN_API_KEY   - Your Respan API key
    OPENAI_API_KEY   - Your OpenAI API key
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from respan_exporter_langchain import RespanCallbackHandler

# 1. Create the Respan callback handler
handler = RespanCallbackHandler(
    api_key="your-respan-api-key",          # or set RESPAN_API_KEY env var
    environment="development",
    customer_identifier="user-123",
    session_identifier="session-abc",
    trace_name="quickstart-chain",
)

# 2. Build a simple prompt -> LLM -> parser chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm | StrOutputParser()

# 3. Invoke the chain with the Respan callback
result = chain.invoke(
    {"question": "What is LangChain in one sentence?"},
    config={"callbacks": [handler]},
)

print(result)
