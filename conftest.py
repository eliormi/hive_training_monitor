"""Root conftest: ensures the project root is on sys.path for all tests,
including Streamlit's AppTest runner which executes scripts in-process."""

import os
import sys

# Add project root to sys.path so `from src.xxx import ...` works everywhere
_PROJECT_ROOT = os.path.dirname(__file__)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
