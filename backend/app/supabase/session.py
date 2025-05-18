"""Initialize Supabase instance."""

import logging

from app.core import settings
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

log = logging.getLogger(__name__)

supabase: Client = create_client(
    settings.supabase.url,
    settings.supabase.key.get_secret_value(),
    options=ClientOptions(
        auto_refresh_token=False,
        persist_session=False,
    ),
)

log.info("Initialized SupaBase SDK")
