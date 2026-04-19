"""
run.py
───────
Windows-safe launcher for HR PolicyBot.
Adds the project root to sys.path before starting uvicorn,
solving the 'No module named app' error on Windows.

Usage:
    python run.py
"""

import sys
import os

# Add project root to Python path — fixes Windows subprocess path issue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
    )