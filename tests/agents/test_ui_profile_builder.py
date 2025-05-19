"""Streamlit-based UI to test the profile builder agent."""

import streamlit as st
import uuid
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from vibe_trader_agent.misc import extract_json     # type: ignore
from vibe_trader_agent.nodes import route_model_output, profile_builder # type: ignore
from vibe_trader_agent.state import InputState, State   # type: ignore
from vibe_trader_agent.tools import TOOLS               # type: ignore

load_dotenv()


def create_profile_builder():
    """Graph consisting of a single node - Profile Builder"""
    builder = StateGraph(State, input=InputState)

    builder.add_node("profile_builder", profile_builder)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "profile_builder")
    builder.add_edge("tools", "profile_builder")
    builder.add_conditional_edges(
        "profile_builder",
        route_model_output,
    )

    # Add memory
    memory = MemorySaver()
    return builder.compile(name="Test Profile Builder", checkpointer=memory)


def initialize_session_state():
    """Initialize session state variables"""
    
    # Session state (the memory bank)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    #  LangGraph Agent (The Brain)
    if "agent" not in st.session_state:
        st.session_state.agent = create_profile_builder()
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False


async def start_conversation_async():
    """Start the conversation with the agent (async version)"""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Start with initial message
    output = await st.session_state.agent.ainvoke(
        {"messages": [HumanMessage(content="Hello! Please be brief and clear in your responses.")]}, 
        config
    )
    
    # Add initial message to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": output['messages'][-1].content
    })
    
    st.session_state.conversation_started = True


def start_conversation():
    """Wrapper to run async start_conversation in sync context"""
    if not st.session_state.conversation_started:
        asyncio.run(start_conversation_async())


async def process_user_message_async(user_input: str):
    """Process user message and get agent response (async version)"""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Get agent response
    input_messages = [HumanMessage(content=user_input)]
    output = await st.session_state.agent.ainvoke({"messages": input_messages}, config)
    
    # Add agent response to chat history
    response_content = output['messages'][-1].content
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content
    })
    
    # Check for extraction completion
    if "EXTRACTION COMPLETE".lower() in response_content.lower():
        extracted_data = extract_json(response_content)
        if extracted_data:
            st.session_state.extracted_data = extracted_data
            st.success("Profile extraction completed!")


def process_user_message(user_input: str):
    """Wrapper to run async process_user_message in sync context"""
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Run async processing
    asyncio.run(process_user_message_async(user_input))


def main():
    st.title("Profile Builder Chat")
    st.caption("Chat with your Profile Builder to build your investment profile")
    
    # Initialize session state
    initialize_session_state()
    
    # Start conversation if not already started
    start_conversation()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process message and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                process_user_message(prompt)
                # Display the latest assistant message
                st.write(st.session_state.messages[-1]["content"])
    
    # Display extracted data if available
    if hasattr(st.session_state, 'extracted_data'):
        with st.expander("📊 Extracted Investment Profile", expanded=True):
            st.json(st.session_state.extracted_data)
    
    # Sidebar with conversation controls
    with st.sidebar:
        st.header("Conversation Controls")
        
        if st.button("🔄 Reset Conversation"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.write(f"**Thread ID:** {st.session_state.thread_id}")
        st.write(f"**Messages:** {len(st.session_state.messages)}")

if __name__ == "__main__":
    main()

