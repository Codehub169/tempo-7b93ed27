#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the port for the Streamlit application
PORT=9000

# Check if requirements.txt exists and install dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "WARNING: requirements.txt not found. Skipping dependency installation."
fi

# Ensure the GOOGLE_API_KEY environment variable is set.
# This variable should be provided by the deployment environment.
# The Streamlit application (app.py) prompts for an API key via its UI, which is then used for API calls.
# This check ensures the environment is properly configured if underlying libraries or other components expect this variable.
if [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: GOOGLE_API_KEY environment variable is not set."
  echo "Please set it in your environment before running the application, e.g., export GOOGLE_API_KEY='your_api_key'"
  exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
  echo "ERROR: app.py not found. Make sure you are in the correct directory."
  exit 1
fi

# Run the Streamlit application
echo "Starting Streamlit application on port $PORT..."
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
