# Respan Tracing + Instructor Integration Tests

This directory contains comprehensive tests demonstrating how to integrate Respan tracing with the [Instructor library](https://github.com/567-labs/instructor) for structured LLM outputs.

## Overview

The Instructor library provides a simple way to get structured data from LLMs using Pydantic models. Respan tracing automatically captures all the LLM calls made through Instructor, providing detailed observability for your structured output workflows.

## Test Files

### 1. `instructor_integration_test.py` - Quick Setup Verification
**Purpose**: Verify that your environment is set up correctly
**Features**:
- ✅ Environment validation
- ✅ Simple structured extraction
- ✅ Basic integration test
- ✅ Setup troubleshooting

**Run this first** to ensure everything is working:
```bash
python tests/tracing-tests/instructor_integration_test.py
```

### 2. `instructor_basic_test.py` - Basic Structured Outputs
**Purpose**: Demonstrate basic structured data extraction patterns
**Features**:
- ✅ Single object extraction (User model)
- ✅ Company information extraction
- ✅ Multiple objects extraction (UserList)
- ✅ Validation and error handling
- ✅ Workflow tracing

**Models demonstrated**:
- `User` - Basic user information
- `CompanyInfo` - Company details
- `UserList` - List of users with metadata

### 3. `instructor_advanced_test.py` - Advanced Features
**Purpose**: Showcase complex validation, retries, and nested models
**Features**:
- ✅ Complex nested Pydantic models
- ✅ Custom validation with automatic retries
- ✅ Chain of thought reasoning
- ✅ Error handling and validation failures
- ✅ Multiple extraction strategies

**Models demonstrated**:
- `Task` - Complex task with validation
- `ProjectSummary` - Nested project information
- `ChainOfThoughtAnalysis` - Structured reasoning

### 4. `instructor_multi_provider_test.py` - Multi-Provider Support
**Purpose**: Show how tracing works across different LLM providers
**Features**:
- ✅ OpenAI GPT models
- ✅ Anthropic Claude models (if configured)
- ✅ Provider comparison
- ✅ Cross-provider result analysis

**Tasks demonstrated**:
- Text sentiment analysis
- Code explanation
- Creative writing

## Prerequisites

### 1. Environment Setup
Make sure you have the required environment variables in your `.env` file:

```bash
# Required
RESPAN_API_KEY=kwai-xxx
OPENAI_API_KEY=sk-xxx

# Optional (for multi-provider tests)
ANTHROPIC_API_KEY=sk-ant-xxx

# Respan Configuration
RESPAN_BASE_URL=http://localhost:8000/api  # or your Respan instance
```

### 2. Virtual Environment
Activate the virtual environment and install dependencies:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install instructor (already done if you followed setup)
poetry add instructor
```

### 3. Dependencies
The following packages are automatically installed:
- `respan-tracing` - Respan tracing SDK
- `instructor` - Structured outputs library
- `openai` - OpenAI client
- `anthropic` - Anthropic client (optional)
- `pydantic` - Data validation

## Running the Tests

### Quick Start
1. **Verify setup** (run this first):
   ```bash
   python tests/tracing-tests/instructor_integration_test.py
   ```

2. **Basic examples**:
   ```bash
   python tests/tracing-tests/instructor_basic_test.py
   ```

3. **Advanced features**:
   ```bash
   python tests/tracing-tests/instructor_advanced_test.py
   ```

4. **Multi-provider comparison**:
   ```bash
   python tests/tracing-tests/instructor_multi_provider_test.py
   ```

### Expected Output
Each test will:
- ✅ Initialize Respan tracing
- ✅ Run structured extraction tasks
- ✅ Display extracted data
- ✅ Validate results
- ✅ Show tracing information

## What Gets Traced

Respan automatically captures:

### LLM Request Details
- Model name and parameters
- Input messages and prompts
- Temperature, max_tokens, etc.
- Provider-specific settings

### Structured Output Information
- Pydantic model schemas
- Validation rules and constraints
- Retry attempts on validation failures
- Parsing and conversion details

### Response Data
- Raw LLM responses
- Structured output results
- Token usage and costs
- Response timing

### Workflow Context
- Task and workflow names
- Execution hierarchy
- Custom attributes and metadata
- Error handling and exceptions

## Viewing Traces

After running any test, check your Respan dashboard to see:

1. **Workflow Traces**: Complete execution flow
2. **LLM Calls**: Individual API requests
3. **Token Usage**: Cost and usage analytics
4. **Validation**: Retry attempts and failures
5. **Performance**: Response times and latency

## Common Patterns

### Basic Extraction
```python
from pydantic import BaseModel
import instructor
from respan_tracing.decorators import task

client = instructor.from_provider("openai/gpt-4o-mini")

class User(BaseModel):
    name: str
    age: int

@task(name="extract_user")
def extract_user(text: str) -> User:
    return client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": text}]
    )
```

### With Validation and Retries
```python
from pydantic import BaseModel, field_validator

class ValidatedUser(BaseModel):
    name: str
    email: str
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

@task(name="extract_with_validation")
def extract_validated_user(text: str) -> ValidatedUser:
    return client.chat.completions.create(
        response_model=ValidatedUser,
        messages=[{"role": "user", "content": text}],
        max_retries=3  # Automatic retries on validation failure
    )
```

### Multi-Provider Setup
```python
# Different providers
openai_client = instructor.from_provider("openai/gpt-4o-mini")
anthropic_client = instructor.from_provider("anthropic/claude-haiku-4-5-20251001")

# Same model, different providers
@task(name="openai_extraction")
def extract_with_openai(text: str) -> User:
    return openai_client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": text}]
    )

@task(name="anthropic_extraction")
def extract_with_anthropic(text: str) -> User:
    return anthropic_client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": text}]
    )
```

## Troubleshooting

### Common Issues

1. **"No module named 'instructor'"**
   ```bash
   poetry add instructor
   ```

2. **"Missing environment variables"**
   - Check your `.env` file
   - Ensure `RESPAN_API_KEY` and `OPENAI_API_KEY` are set

3. **"Validation failed"**
   - Check your Pydantic model constraints
   - Use `max_retries` parameter for automatic retries
   - Review validation error messages

4. **"Anthropic not available"**
   - Set `ANTHROPIC_API_KEY` in `.env`
   - Tests will skip Anthropic if not configured

### Debug Mode
Enable debug logging to see detailed trace information:

```python
from respan_tracing import RespanTelemetry

k_tl = RespanTelemetry(
    app_name="debug-test",
    log_level="DEBUG"
)
```

## Integration with Existing Code

To add Respan tracing to your existing Instructor code:

1. **Add tracing initialization**:
   ```python
   from respan_tracing import RespanTelemetry
   k_tl = RespanTelemetry(app_name="my-app")
   ```

2. **Add decorators to your functions**:
   ```python
   from respan_tracing.decorators import task, workflow
   
   @task(name="my_extraction")
   def my_function():
       # Your existing Instructor code
       pass
   ```

3. **That's it!** - All LLM calls are automatically traced

## Next Steps

1. **Run the integration test** to verify setup
2. **Explore the example tests** to understand patterns
3. **Adapt the patterns** to your use case
4. **Check the Respan dashboard** for traces
5. **Integrate with your existing code**

## Resources

- [Instructor Documentation](https://python.useinstructor.com/)
- [Respan Tracing Documentation](../README.md)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Anthropic API Documentation](https://docs.anthropic.com/)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the integration test output
3. Check your Respan dashboard for error details
4. Ensure all environment variables are correctly set