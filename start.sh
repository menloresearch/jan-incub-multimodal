#!/bin/bash

# Disable large models by default to save resources
export DISABLED_MODELS="large-v1,large-v2,large-v3,large"

uv run run_server.py
