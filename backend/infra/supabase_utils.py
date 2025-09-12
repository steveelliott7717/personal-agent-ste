# backend/infra/supabase_utils.py
import os
from supabase import create_client

_supa = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE"])


def sign_export_path(path: str, expires_in: int = 3600) -> str:
    res = _supa.storage.from_("chatgpt_exports").create_signed_url(
        path, expires_in=expires_in
    )
    return res.get("signedURL") or res.get("signed_url")
