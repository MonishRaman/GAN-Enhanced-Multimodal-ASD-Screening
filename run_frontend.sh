#!/usr/bin/env bash
echo "========================================================"
echo "Starting ASD Multimodal Analysis Frontend Server"
echo "URL: http://127.0.0.1:5500"
echo "========================================================"
python -m http.server 5500 --directory frontend

