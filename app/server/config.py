"""Dual-mode authentication: Databricks App vs local."""
import os
from databricks.sdk import WorkspaceClient

IS_DATABRICKS_APP = bool(os.environ.get("DATABRICKS_APP_NAME"))

CATALOG = os.environ.get("CATALOG", "your_catalog")
SCHEMA = os.environ.get("SCHEMA", "agent_eval")
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"
LLM_ENDPOINT = os.environ.get("SERVING_ENDPOINT", "databricks-claude-sonnet-4-6")
VS_ENDPOINT = os.environ.get("VS_ENDPOINT", "your_vs_endpoint")
VS_INDEX = f"{FULL_SCHEMA}.knowledge_base_index"


def get_workspace_client() -> WorkspaceClient:
    if IS_DATABRICKS_APP:
        return WorkspaceClient()
    profile = os.environ.get("DATABRICKS_PROFILE", "DEFAULT")
    return WorkspaceClient(profile=profile)


def get_oauth_token() -> str:
    client = get_workspace_client()
    auth = client.config.authenticate()
    if isinstance(auth, dict) and "Authorization" in auth:
        return auth["Authorization"].replace("Bearer ", "")
    return getattr(client.config, 'token', '') or ''


def get_workspace_host() -> str:
    if IS_DATABRICKS_APP:
        host = os.environ.get("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            host = f"https://{host}"
        return host
    client = get_workspace_client()
    return client.config.host
