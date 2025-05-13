import asyncio
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()

from vibe_trader_agent.nodes import views_analyst, route_model_output
from vibe_trader_agent.state import State, InputState
from vibe_trader_agent.tools import search_market_data
from vibe_trader_agent.finance_tools import calculate_financial_metrics
from vibe_trader_agent.misc import extract_json
from vibe_trader_agent.tools import TOOLS


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
    state.messages = [HumanMessage(content="[IBM, BTC]")]
    response = await graph.ainvoke(state, {"recursion_limit": 10})

    print(response)


# Run the agent test
if __name__ == "__main__":
    asyncio.run(main())


# # Create the State
# state = State()
# state.messages = [HumanMessage(content="[IBM, ABNB]")]

# # Run the async function directly
# response = asyncio.run(views_analyst(state))
# print(response)

# result = {}
# if isinstance(response.content, str) and "EXTRACTION COMPLETE".lower() in response.content.lower():        
#     extracted_data = extract_json(response.content)
#     result["views_created"] = extracted_data if extracted_data else {"error": "missing generated views"}

# print("\n\n")
# print(result)
