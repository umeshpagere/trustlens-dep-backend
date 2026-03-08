"""
TrustLens entry point (Production - Render)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import create_app

app = create_app()
