import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
print('[supabase] URL =', SUPABASE_URL)
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE")


from supabase import create_client
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)