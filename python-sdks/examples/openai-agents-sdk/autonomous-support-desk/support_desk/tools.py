"""Tools = decisions = filterable breadcrumbs (plan §4.2).

Every terminal action is a tool, so the agent's autonomy leaves structured,
filterable spans. Terminal tools set ctx.resolution (a STRING ENUM - never a
boolean, plan §3-item-6).

  search_kb       -> embeds the query (embedding span) + returns KB chunks
  lookup_account  -> mock account data
  lookup_order    -> mock order data
  process_refund  -> terminal: resolution = refunded
  close_resolved  -> terminal: resolution = self_resolved
  escalate_to_human -> terminal: resolution = escalated
"""

from agents import RunContextWrapper, custom_span, function_tool

from .config import KB, MOCK_ACCOUNTS, MOCK_ORDERS
from .context import TicketContext
from .telemetry import get_gateway_client

EMBED_MODEL = "text-embedding-3-small"


def _retrieve(tenant_id: str, query: str, k: int = 2) -> list[dict]:
    """Tiny keyword-overlap retrieval over the tenant KB (mock vector search)."""
    terms = {t for t in query.lower().split() if len(t) > 3}
    corpus = KB.get(tenant_id, [])
    scored = []
    for chunk in corpus:
        text = (chunk["title"] + " " + chunk["body"]).lower()
        score = sum(1 for t in terms if t in text)
        scored.append((score, chunk))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [c for score, c in scored[:k] if score > 0] or [corpus[0]] if corpus else []


@function_tool
async def search_kb(ctx: RunContextWrapper[TicketContext], query: str) -> dict:
    """Search the knowledge base for an answer to the customer's question."""
    # Real embedding call through the gateway. Wrap it in an Agents-SDK span so it
    # nests under the ticket's trace (the live Agents SDK trace context) instead of
    # orphaning into its own single-span trace.
    client = get_gateway_client()
    with custom_span("openai.embeddings", data={"model": EMBED_MODEL}):
        await client.embeddings.create(model=EMBED_MODEL, input=query)
    chunks = _retrieve(ctx.context.tenant_id, query)
    ctx.context.record("search_kb", query=query, hits=len(chunks))
    return {"query": query, "results": [{"title": c["title"], "body": c["body"]} for c in chunks]}


@function_tool
def lookup_account(ctx: RunContextWrapper[TicketContext], customer_id: str) -> dict:
    """Look up the customer's account record."""
    ctx.context.record("lookup_account", customer_id=customer_id)
    return MOCK_ACCOUNTS.get(customer_id, {"customer_id": customer_id, "found": False})


@function_tool
def lookup_order(ctx: RunContextWrapper[TicketContext], order_id: str) -> dict:
    """Look up an order by its id (e.g. A-1001)."""
    ctx.context.record("lookup_order", order_id=order_id)
    return MOCK_ORDERS.get(order_id, {"order_id": order_id, "found": False})


@function_tool
def process_refund(ctx: RunContextWrapper[TicketContext], order_id: str, amount: float) -> dict:
    """Issue a refund for an order. Terminal action - resolves the ticket as refunded."""
    ctx.context.resolution = "refunded"
    ctx.context.record("process_refund", order_id=order_id, amount=amount)
    return {"order_id": order_id, "amount": amount, "status": "refunded"}


@function_tool
def close_resolved(ctx: RunContextWrapper[TicketContext], summary: str) -> dict:
    """Close the ticket as resolved directly. Terminal action - resolution self_resolved."""
    ctx.context.resolution = "self_resolved"
    ctx.context.record("close_resolved", summary=summary)
    return {"status": "resolved", "summary": summary}


@function_tool
def escalate_to_human(ctx: RunContextWrapper[TicketContext], reason: str) -> dict:
    """Escalate the ticket to a human agent. Terminal action - resolution escalated."""
    ctx.context.resolution = "escalated"
    ctx.context.record("escalate_to_human", reason=reason)
    return {"status": "escalated", "reason": reason}
