# Vibe Trader Agent

A sophisticated AI agent built with LangGraph that combines financial advisory capabilities with personalized user profiling. The agent uses a state-based graph architecture to manage conversations and provide tailored financial advice.

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

3. Setup Env and install Dependencies
```bash
./taskfile setup_venv
```

4. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Project Structure

```
vibe-trader-agent/
├── src/
│   └── vibe_trader_agent/
│       ├── graph.py          # Main graph definition
│       ├── nodes.py          # Node implementations
│       ├── state.py          # State management
│       ├── tools.py          # Tool definitions
│       └── configuration.py  # Configuration settings
├── tests/                    # Test suite
├── static/                   # Static assets
└── pyproject.toml           # Project configuration
```

## Usage

The agent can be used as a Python package:

```python
from vibe_trader_agent.graph import graph

# Initialize the agent
agent = graph.compile()

# Run the agent
result = agent.invoke({"input": "Hello, I need financial advice"})
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Run linting:
```bash
ruff check .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Inspired by the need for personalized financial advisory systems

https://github.com/tevslin/meeting-reporter/blob/main/mm_agent.py

