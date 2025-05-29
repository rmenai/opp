"""Initialize Supabase instance."""

import logging

from fastapi import HTTPException

from app.core import settings
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

log = logging.getLogger(__name__)


def create_supabase() -> Client:
    """Create fresh Supabase client instance."""
    client: Client = create_client(
        settings.supabase.url,
        settings.supabase.key.get_secret_value(),
        options=ClientOptions(
            auto_refresh_token=False,
            persist_session=False,
        ),
    )

    if not client:
        raise HTTPException(status_code=500, detail="Super client not initialized")

    log.info("Initialized new Supabase SDK instance")
    return client
