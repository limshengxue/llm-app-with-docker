#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for Ollama server to start
sleep 5

# Pull model
ollama pull mistral

# Wait for the Ollama server to finish 
wait $!