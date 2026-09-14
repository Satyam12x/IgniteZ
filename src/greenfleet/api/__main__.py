"""Run the platform: ``python -m greenfleet.api [--host H] [--port P] [--reload]``."""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    uvicorn.run("greenfleet.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
