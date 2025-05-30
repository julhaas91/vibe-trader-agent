#!/bin/bash
set -e

# Tasks
run() {
  uv run langgraph dev --allow-blocking
}

setup_venv() {
  python3 -m venv .venv
  echo "... created new virtual environment ..."
  source .venv/bin/activate
  echo "... activated new virtual environment ..."
  uv sync
  source .env
}

reset_venv() {
  sudo rm -rf .venv
  echo "... deleted old virtual environment ..."
  python3 -m venv .venv
  echo "... created new virtual environment ..."
  source .venv/bin/activate
  echo "... activated new virtual environment ..."
  uv sync --frozen
  source .env
}

# Check if the provided argument matches any of the functions
if declare -f "$1" > /dev/null; then
  "$@"  # If the function exists, run it with any additional arguments
else
  echo "Error: Unknown task '$1'"
  echo
  help  # Show help if the task is not recognized
fi

# Run application if no argument is provided
"${@:-run}"
