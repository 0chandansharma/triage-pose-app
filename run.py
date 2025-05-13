#!/usr/bin/env python
"""
Run script for Triage-Pose application
"""
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Change to the triage-pose directory
os.chdir(os.path.join(project_root, 'triage-pose'))

# Start the uvicorn server
import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)