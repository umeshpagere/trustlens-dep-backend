"""
TrustLens entry point (Phase 6 async refactor)

Switches from Werkzeug (WSGI) to Hypercorn (ASGI).

Why Hypercorn instead of `app.run()`:
  `app.run()` uses the Werkzeug development server which is WSGI-only.
  WSGI is a synchronous protocol — it blocks on each request, so even
  though the route is marked `async def`, Werkzeug runs it synchronously.
  All concurrency gains from asyncio.gather() are silently lost.

  Hypercorn is a production-grade ASGI server. It runs each request as
  a proper coroutine on the event loop, allowing asyncio.gather() to
  execute phases 2/3/4 concurrently as intended.

Running:
    python run.py
  or directly:
    hypercorn run:app --bind 0.0.0.0:5000
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import create_app
from app.config.settings import Config

app = create_app()


async def _serve() -> None:
    import hypercorn.asyncio
    from hypercorn.config import Config as HypercornConfig

    config = HypercornConfig()
    config.bind = [f"0.0.0.0:{Config.PORT}"]
    config.loglevel = "info"

    print(f"🚀 TrustLens backend running on port {Config.PORT} (Hypercorn ASGI)")
    await hypercorn.asyncio.serve(app, config)


if __name__ == "__main__":
    asyncio.run(_serve())
