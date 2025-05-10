import logging

from app.core import settings
from supabase import Client, create_client

log = logging.getLogger(__name__)

supabase: Client = create_client(settings.supabase.url, settings.supabase.key.get_secret_value())

log.info("Initialized SupaBase SDK")
