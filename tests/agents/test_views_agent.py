import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from vibe_trader_agent.misc import extract_json
from vibe_trader_agent.nodes import route_model_output, views_analyst
from vibe_trader_agent.state import InputState, State
from vibe_trader_agent.tools import TOOLS

load_dotenv()


async def main():
    # Define a graph with individual views agent
    builder = StateGraph(State, input=InputState)

    builder.add_node("views_analyst", views_analyst)
    # builder.add_node("tools", ToolNode([search_market_data, calculate_financial_metrics]))
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "views_analyst")
    builder.add_edge("tools", "views_analyst")
    builder.add_conditional_edges(
        "views_analyst",
        route_model_output,
    )

    graph = builder.compile(name="Vibe Trader Agent")

    # Run the graph
    state = State()
    state.messages = [HumanMessage(content="[SPY, BTC]")]
    response = await graph.ainvoke(state, {"recursion_limit": 10})

    last_msg = response['messages'][-1]
    
    # Extract
    result = {}
    # Check if the response contains the extraction completion marker
    if isinstance(last_msg.content, str) and "EXTRACTION COMPLETE".lower() in last_msg.content.lower():        
        extracted_data = extract_json(last_msg.content)
        result["views_created"] = extracted_data if extracted_data else {"error": "missing generated views"}


# Run the agent test
if __name__ == "__main__":
    asyncio.run(main())

