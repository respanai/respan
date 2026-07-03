"""Autonomous multi-tenant support desk on the OpenAI Agents SDK, instrumented with Respan.

See respan-dev-stuff/respan-support-agent-demo-plan.md for the full design.
This package is the runnable spine (build-sequence steps 1-8): config, tools,
agents, run_ticket, and the concurrent multi-tenant runner.
"""
