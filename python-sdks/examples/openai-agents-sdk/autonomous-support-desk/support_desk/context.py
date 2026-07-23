"""The per-ticket shared run context (plan §4.3).

Passed into Runner.run(..., context=ctx) so terminal tools can record the
outcome and every tool can leave a breadcrumb. After the run, ctx.resolution
is the emergent outcome we stamp onto the trace at trace.finish().
"""

from dataclasses import dataclass, field


@dataclass
class TicketContext:
    tenant_id: str
    ticket_id: str
    resolution: str | None = None       # set by terminal tools: self_resolved | refunded | escalated
    tools_used: list[str] = field(default_factory=list)
    # Lightweight event log so a UI can attach later (plan §9). The Python
    # pipeline is the spine; this is the seam for a future event stream.
    events: list[dict] = field(default_factory=list)

    def record(self, tool: str, **detail) -> None:
        self.tools_used.append(tool)
        self.events.append({"tool": tool, **detail})
