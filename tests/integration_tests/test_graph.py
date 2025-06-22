import pytest
from langsmith import unit

from vibe_trader_agent import graph
from vibe_trader_agent.prompts import PROFILER_SYSTEM_PROMPT


@pytest.mark.asyncio
@unit
async def test_vibe_trader_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},
        {"configurable": {"system_prompt": PROFILER_SYSTEM_PROMPT}},
    )

    assert "harrison" in str(res["messages"][-1].content).lower()
