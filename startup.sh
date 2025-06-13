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

# Set the Google API Key from the environment variable provided during startup, or use a default if not set
# The API key provided by the user during the interaction will be used here.
export GOOGLE_API_KEY="AIzaSyCAH4QbvHnjo4hQXKMAhaI9KP8gr3WVMB4"

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: GOOGLE_API_KEY environment variable is not set."
  echo "Please set it before running the application, e.g., export GOOGLE_API_KEY='your_api_key'"
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
