"""Research Bot — Multi-agent research workflow with parallel web search."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Union
from dotenv import load_dotenv

load_dotenv(override=True)

from pydantic import BaseModel
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner

from respan import Respan
from respan_instrumentation_openai_agents import OpenAIAgentsInstrumentor
from agents import Agent, Runner, WebSearchTool, custom_span, gen_trace_id, trace
from agents.model_settings import ModelSettings

respan = Respan(instrumentations=[OpenAIAgentsInstrumentor()])


# --- Models ---

class WebSearchItem(BaseModel):
    reason: str
    query: str


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]


class ReportData(BaseModel):
    short_summary: str
    markdown_report: str
    follow_up_questions: list[str]


# --- Agents ---

planner_agent = Agent(
    name="PlannerAgent",
    instructions=(
        "You are a helpful research assistant. Given a query, come up with a set of web searches "
        "to perform to best answer the query. Output between 5 and 20 terms to query for."
    ),
    model="gpt-4o",
    output_type=WebSearchPlan,
)

search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You are a research assistant. Given a search term, search the web and produce a concise "
        "summary of the results. The summary must be 2-3 paragraphs and less than 300 words."
    ),
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
)

writer_agent = Agent(
    name="WriterAgent",
    instructions=(
        "You are a senior researcher. Write a cohesive report for a research query. "
        "The final output should be in markdown format, at least 1000 words."
    ),
    model="o3-mini",
    output_type=ReportData,
)


# --- Printer ---

class Printer:
    def __init__(self, console: Console):
        self.live = Live(console=console)
        self.items: dict[str, tuple[str, bool]] = {}
        self.hide_done_ids: set[str] = set()
        self.live.start()

    def end(self) -> None:
        self.live.stop()

    def update_item(self, item_id: str, content: str, is_done: bool = False, hide_checkmark: bool = False) -> None:
        self.items[item_id] = (content, is_done)
        if hide_checkmark:
            self.hide_done_ids.add(item_id)
        self.flush()

    def mark_item_done(self, item_id: str) -> None:
        self.items[item_id] = (self.items[item_id][0], True)
        self.flush()

    def flush(self) -> None:
        renderables: list[Any] = []
        for item_id, (content, is_done) in self.items.items():
            if is_done:
                prefix = "  " if item_id not in self.hide_done_ids else ""
                renderables.append(prefix + content)
            else:
                renderables.append(Spinner("dots", text=content))
        self.live.update(Group(*renderables))


# --- Manager ---

class ResearchManager:
    def __init__(self):
        self.console = Console()
        self.printer = Printer(self.console)

    async def run(self, query: str) -> None:
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            self.printer.update_item("starting", "Starting research...", is_done=True, hide_checkmark=True)
            search_plan = await self._plan_searches(query)
            search_results = await self._perform_searches(search_plan)
            report = await self._write_report(query, search_results)
            self.printer.update_item("final_report", f"Report summary\n\n{report.short_summary}", is_done=True)
            self.printer.end()

        print("\n\n=====REPORT=====\n\n")
        print(report.markdown_report)
        print("\n\n=====FOLLOW UP QUESTIONS=====\n\n")
        print("\n".join(report.follow_up_questions))

    async def _plan_searches(self, query: str) -> WebSearchPlan:
        self.printer.update_item("planning", "Planning searches...")
        result = await Runner.run(planner_agent, f"Query: {query}")
        self.printer.update_item("planning", f"Will perform {len(result.final_output.searches)} searches", is_done=True)
        return result.final_output_as(WebSearchPlan)

    async def _perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        with custom_span("Search the web"):
            self.printer.update_item("searching", "Searching...")
            tasks = [asyncio.create_task(self._search(item)) for item in search_plan.searches]
            results, num_completed = [], 0
            for task in asyncio.as_completed(tasks):
                result = await task
                if result is not None:
                    results.append(result)
                num_completed += 1
                self.printer.update_item("searching", f"Searching... {num_completed}/{len(tasks)} completed")
            self.printer.mark_item_done("searching")
            return results

    async def _search(self, item: WebSearchItem) -> Union[str, None]:
        try:
            result = await Runner.run(search_agent, f"Search term: {item.query}\nReason: {item.reason}")
            return str(result.final_output)
        except Exception:
            return None

    async def _write_report(self, query: str, search_results: list[str]) -> ReportData:
        self.printer.update_item("writing", "Thinking about report...")
        result = Runner.run_streamed(writer_agent, f"Original query: {query}\nSearch results: {search_results}")
        update_messages = ["Planning structure...", "Writing outline...", "Creating sections...", "Finalizing..."]
        last_update, next_msg = time.time(), 0
        async for _ in result.stream_events():
            if time.time() - last_update > 5 and next_msg < len(update_messages):
                self.printer.update_item("writing", update_messages[next_msg])
                next_msg += 1
                last_update = time.time()
        self.printer.mark_item_done("writing")
        return result.final_output_as(ReportData)


async def main():
    query = "What are the latest developments in quantum computing?"
    await ResearchManager().run(query)
    respan.flush()


if __name__ == "__main__":
    asyncio.run(main())
