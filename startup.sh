#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the port for the Streamlit application
PORT=9000

# Check if requirements.txt exists and install dependencies
if [ -f "requirements.txt" ]; then
    echo "requirements.txt found. Installing dependencies..."
    pip install --no-cache-dir -r requirements.txt
else
    echo "WARNING: requirements.txt not found. Skipping dependency installation."
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found. Cannot start application."
    exit 1
fi

# Run the Streamlit application
echo "Starting Streamlit application on port $PORT..."
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
