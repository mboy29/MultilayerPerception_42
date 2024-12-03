#!/bin/sh

# GLOBALS
# -------

# Define ANSI color escape codes
ERROR='\033[38;2;255;173;194m'
INFO='\033[38;2;204;219;253m'
SUCCESS='\033[38;2;207;225;185m'
NC='\033[0m'  # No Color

# Define variables
PYTHON_PATH="/usr/bin/python3.10"
VENV_PATH="./venv"


# FUNCTIONS
# ---------

# Function to deactivate and remove the virtual environment
deactivate_venv() {
    local DEACT=$1
    local VENV_EXISTS=false
    
    echo "${INFO}virtualenv.sh: Deactivating virtual env - ${VENV_PATH}...${NC}"
    
    # Check if we are in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate || true
        VENV_EXISTS=true
    fi
    
    rm -rf "$VENV_PATH"
    echo "${SUCCESS}virtualenv.sh: Virtual environment removed!${NC}"
    if [ "$DEACT" = false ] && [ "$VENV_EXISTS" = true ]; then
        echo ""
    fi
}

# Function to create and activate the virtual environment, and install requirements
activate_venv() {
    echo "${INFO}virtualenv.sh: Creating virtual env ${VENV_PATH}...${NC}"
    python3 -m venv "$VENV_PATH" > /dev/null 2>&1
    source "$VENV_PATH/bin/activate"

    echo "${INFO}virtualenv.sh: Upgrading pip, setuptools, and wheel...${NC}"
    python3 -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    echo "${INFO}virtualenv.sh: Installing requirements...${NC}"
    python3 -m pip install -r requirements.txt --no-cache-dir --force-reinstall > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "${SUCCESS}virtualenv.sh: Requirements installed successfully!${NC}"
    else
        echo "${ERROR}virtualenv.sh: Failed to install requirements. Check the logs above.${NC}"
        deactivate_venv true
        return 1
    fi

    echo "${SUCCESS}virtualenv.sh: Virtual environment is ready!${NC}"
}


# SCRIPT
# ------

# Main script logic
if [ "$1" = "--activate" ] || [ $# -eq 0 ]; then
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate_venv false
        activate_venv
    else
        activate_venv
    fi
elif [ "$1" = "--deactivate" ]; then
    deactivate_venv true
else
    echo "${ERROR}virtualenv.sh: Invalid argument: $1${NC}"
    echo "${INFO}virtualenv.sh: source ./virtualenv.sh [--activate] [--deactivate]${NC}"
    echo "${INFO} - Use --activate or no argument to activate virtual env.${NC}"
    echo "${INFO} - Use --deactivate to deactivate virtual env.${NC}"
fi
