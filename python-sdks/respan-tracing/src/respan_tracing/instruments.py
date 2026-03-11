from enum import Enum


class Instruments(Enum):
    """Enumeration of available instrumentation libraries"""

    # AI/ML Libraries
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    GROQ = "groq"
    TOGETHER = "together"
    REPLICATE = "replicate"
    TRANSFORMERS = "transformers"

    # Cloud AI Services
    BEDROCK = "bedrock"
    SAGEMAKER = "sagemaker"
    VERTEXAI = "vertexai"
    GOOGLE_GENERATIVEAI = "google_generativeai"
    WATSONX = "watsonx"
    ALEPHALPHA = "alephalpha"

    # Vector Databases
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"
    LANCEDB = "lancedb"
    MARQO = "marqo"

    # Frameworks
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"
    HAYSTACK = "haystack"
    CREW = "crew"
    MCP = "mcp"

    # Infrastructure
    CELERY = "celery"
    DJANGO = "django"
    FASTAPI = "fastapi"
    FLASK = "flask"
    SQLALCHEMY = "sqlalchemy"
    PSYCOPG2 = "psycopg2"
    AIOHTTP_CLIENT = "aiohttp_client"
    GRPC = "grpc"
    REDIS = "redis"
    REQUESTS = "requests"
    URLLIB3 = "urllib3"
    PYMYSQL = "pymysql"
    THREADING = "threading"  # Context propagation across threads
