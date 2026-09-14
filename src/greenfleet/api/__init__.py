"""Phase 4: the decision-support platform.

A FastAPI service over the Phase 1-3 engines, plus a single-page dashboard served
from ``static/``. Nothing here re-implements a calculation: every number the API
returns comes from the same modules the tests assert on.
"""
