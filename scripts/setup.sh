#!/bin/bash
# Setup script for Student Records Chatbot
# Usage: ./scripts/setup.sh

set -e

echo "======================================"
echo "Student Records Chatbot Setup"
echo "======================================"

# Check for UV
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "UV version: $(uv --version)"

# Create virtual environment
echo "Creating virtual environment..."
uv venv .venv --python 3.11

# Activate
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Create environment file if not exists
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "Please edit .env with your API keys"
fi

# Create data directories
mkdir -p data/raw data/processed

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
echo "3. Run the application:"
echo "   - CLI: python -m src.agent"
echo "   - API: uvicorn src.api:app --reload"
echo "   - LangGraph Dev: langgraph dev"
