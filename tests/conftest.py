"""Shared test setup. Runs before any test module imports metricmedic."""

import os

# Tests must never send traces to Langfuse, even if keys are in .env.
# load_dotenv() does not override variables that are already set.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
