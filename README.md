# Vibe Trader Agent

A sophisticated AI agent built with LangGraph that combines financial advisory capabilities with personalized user profiling. The agent uses a state-based graph architecture to manage conversations and provide tailored financial advice.

**Vibe-Trader** employs a multi-agent system with specialized AI agents that work together to deliver personalized investment recommendations:
![Architecture Diagram](static/multi-agent-architecture.png)

## 🎥 Walkthrough Video

📺 **[Watch the Vibe Trader Agent Walkthrough](https://drive.google.com/file/d/1Hh8aWtkRbw2lZFXaqOQuhcEP5U65WahW/view?usp=sharing)**

Get a comprehensive overview of how the Vibe Trader Agent works, including demonstrations of the multi-agent system, user profiling, and portfolio optimization features.

## Graph Nodes

| Node | Role | Description |
|-------|------|-------------|
| **Profiler** | Investment DNA Capture | Engages in natural conversation to build your complete financial profile, gathering risk tolerance, timeline, goals, and investment preferences |
| **Mandate Strategist** | Investment Rules Creation | Develops your personalized investment mandate, identifying what to invest in, what to avoid, and establishing your unique investment constraints |
| **Asset Researcher** | Market Intelligence | Scans thousands of investment opportunities to handpick 10-15 assets that perfectly match your profile while ensuring proper diversification |
| **Portfolio Analyst** | Risk-Return Analysis | Analyzes market views and runs advanced Monte Carlo simulations to evaluate portfolio performance and risk characteristics |
| **Portfolio Optimizer** | Allocation Optimization | Calculates optimal asset allocation to maximize your probability of reaching financial goals using quantitative optimization models |
| **Reporter** | Professional Documentation | Generates comprehensive investment reports with clear insights, portfolio composition, and actionable recommendations |

The agents work sequentially, with each building upon the previous agent's output to create a comprehensive, personalized investment strategy tailored specifically to your financial situation and goals.


## Features

- **Conversational AI** that adapts to your expertise level
- **Human-in-the-Loop** verification for critical decisions
- **Tool Integration**: Supports various tools for enhanced functionality
- **Market Data Integration** from multiple financial APIs
- **State Management**: Robust state management for maintaining conversation context
- **Interruptible Workflows**: Flexible conversation flow with support for tool interruptions
- **Professional Reporting** with data visualization and cloud storage


## Requirements

- Python 3.11 or higher
- OpenAI API key (or other supported LLM provider)
- Other dependencies as specified in `pyproject.toml`
- Other API keys as specified in `.env.example`

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

## Utils
### Google storage
Set google environmental variables for google, see example in .env.example

## How to use google cloud storage
```python 
from utils import init_storage_client, upload_pdf
import os
from dotenv import load_dotenv
load_dotenv('../../.env')

from io import BytesIO
from reportlab.pdfgen import canvas


def generate_pdf() -> bytes:
    """Generate a dummy PDF for example purposes."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(100, 100, "Generated PDF Content")
    pdf.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# Initialize client
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
if not project_id:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")

gcs_client = init_storage_client(project_id=project_id)
content_pdf_bytes = generate_pdf()

# Upload with signed URL (valid for 7 days)
url = upload_pdf(
    client=gcs_client,
    bucket='vibe-trader-reports-dev',
    destination="test_pdf.pdf",
    content=content_pdf_bytes,
    make_public=True,
    expiration_days=7
)
print(f"File uploaded successfully. Access URL: {url}")
```
Using this example u can upload any object converting into bytes stream.
If you specify make_public - as output you will get link to uploaded file.

## Questions and Support

> If you have any questions or need technical support, feel free to reach out.  
> You can contact me via [email](mailto:juliushaas91@gmail.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/jh91/).
