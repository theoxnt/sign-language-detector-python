#!/bin/bash
# Quick run script for Sign Language Detector

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Sign Language Detector${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import cv2, mediapipe, sklearn, torch" 2>/dev/null; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Run the CLI
echo -e "${GREEN}Starting Sign Language Detector...${NC}"
echo ""
python -m src.cli_enhanced "$@"
