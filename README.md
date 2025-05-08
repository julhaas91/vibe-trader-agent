# Vibe Trader Agent

A sophisticated AI agent built with LangGraph that combines financial advisory capabilities with personalized user profiling. The agent uses a state-based graph architecture to manage conversations and provide tailored financial advice.

![Architecture Diagram](static/architecture.png)

## Features

- **Interactive Profile Building**: Dynamically builds user profiles through natural conversation
- **Financial Advisory**: Provides personalized financial advice based on user profiles
- **Tool Integration**: Supports various tools for enhanced functionality
- **State Management**: Robust state management for maintaining conversation context
- **Interruptible Workflows**: Flexible conversation flow with support for tool interruptions

## Requirements

- Python 3.11 or higher
- OpenAI API key (or other supported LLM provider)
- Other dependencies as specified in `pyproject.toml`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vibe-trader-agent.git
cd vibe-trader-agent
```

2. Setup Env and install Dependencies
```bash
./taskfile.sh setup_venv
```

3. Run application locally
```bash
./taskfile.sh run
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Inspired by the need for personalized financial advisory systems

https://github.com/tevslin/meeting-reporter/blob/main/mm_agent.py

