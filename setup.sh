#!/bin/bash
# setup.sh - Streamlit Cloud startup script
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
