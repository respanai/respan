# Prompts Reference

Create, version, deploy, and use prompt templates from the Respan platform.

Docs: `https://respan.ai/docs/documentation/features/prompt-management/prompt-management-quickstart.md`

---

## Overview

Prompts are version-controlled templates stored on the Respan platform. The workflow is:

1. **Create** a prompt with a name
2. **Add versions** with messages, model config, and `{{variables}}`
3. **Deploy** the version you want live
4. **Fetch** the prompt in code using the API or MCP tools
5. **Iterate** — commit new versions and deploy without code changes

---

## Create a Prompt

### Via MCP

```
Use create_prompt to create a prompt:
  name: "classify_intent"
  description: "Classifies user messages into intent categories"

Then use create_prompt_version to add content:
  prompt_id: "<id from above>"
  model: "gpt-4o-mini"
  messages:
    - role: "system"
      content: "You are an intent classifier. Classify the user message into one of: question, complaint, request, feedback."
    - role: "user"
      content: "{{user_message}}"
```

### Via CLI

```bash
respan prompts create --name "classify_intent"
respan prompts versions <prompt-id>
```

---

## Variables

Use `{{variable_name}}` in message content. Variables are filled at runtime.

```json
{
  "messages": [
    {"role": "system", "content": "You are a {{role}} assistant."},
    {"role": "user", "content": "{{user_input}}"}
  ]
}
```

### Jinja templates

Prompts support Jinja2 syntax:

- **Conditionals:** `{% if context %}Use this context: {{context}}{% endif %}`
- **JSON access:** `{{ input.key }}`
- **Filters:** `{{ var | upper }}`
- **Loops:** `{% for item in items %}{{item}}{% endfor %}`
- **Default values:** Set via the API or platform UI

---

## Versioning and Deployment

Each prompt can have multiple versions. Only one version is **deployed** (live) at a time.

### Workflow

1. `create_prompt_version` — creates a new version (always starts as NOT deployed)
2. Test the version in the playground or via API
3. Deploy via platform UI when ready
4. Code using `prompt_id` automatically gets the deployed version

### Version pinning

```python
# Always use the deployed version (default)
prompt = get_prompt(prompt_id="abc123")

# Pin to a specific version
prompt = get_prompt(prompt_id="abc123", version=3)
```

---

## Using Prompts in Code

### Python (via Gateway)

Fetch the prompt config and pass to the LLM:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["RESPAN_API_KEY"],
    base_url=os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api"),
)

# The gateway resolves the prompt template and fills variables
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[],  # Messages come from the prompt template
    extra_body={
        "prompt_id": "your-prompt-id",
        "prompt_variables": {
            "user_message": "I want to return my order",
        },
    },
)
```

### Linking traces to prompts

Use `propagate_attributes` to tag traces with the prompt used:

```python
from respan import propagate_attributes

with propagate_attributes(
    prompt={"prompt_id": "abc123", "variables": {"user_message": "Hello"}},
):
    response = client.chat.completions.create(...)
```

---

## Prompt Composition

Variables can reference other prompts (child prompts rendered first, injected as text).

Configure in the platform UI by changing a variable's type from **Text** to **Prompt**.

Max nesting depth: 2 (parent -> child).

---

## Structured Output (JSON Schema)

Attach a JSON schema to a prompt version for structured responses:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "extraction",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "intent": {"type": "string"},
          "confidence": {"type": "number"}
        },
        "required": ["intent", "confidence"],
        "additionalProperties": false
      }
    }
  }
}
```

---

## MCP Tools Reference

| Action | Tool |
|--------|------|
| List all prompts | `list_prompts` |
| Get prompt details | `get_prompt_detail(prompt_id)` |
| Create a prompt | `create_prompt(name, description)` |
| Update name/description | `update_prompt(prompt_id, name, description)` |
| Create a version | `create_prompt_version(prompt_id, messages, model, ...)` |
| Update a version | `update_prompt_version(prompt_id, version, ...)` |
| List versions | `list_prompt_versions(prompt_id)` |
| Get version detail | `get_prompt_version_detail(prompt_id, version)` |

---

## CLI Commands

```bash
respan prompts list                    # List all prompts
respan prompts get <id>                # Get prompt details
respan prompts create --name <name>    # Create a prompt
respan prompts update <id>             # Update a prompt
respan prompts versions <id>           # List prompt versions
```
