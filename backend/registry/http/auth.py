import base64, hmac, hashlib, os


def apply_auth_headers(req, auth: dict):
    atype = (auth.get("type") or "").lower()
    if atype == "bearer" and auth.get("token"):
        req.headers.setdefault("Authorization", f"Bearer {auth['token']}")
    elif atype == "basic" and auth.get("username") and auth.get("password"):
        b64 = base64.b64encode(
            f"{auth['username']}:{auth['password']}".encode()
        ).decode()
        req.headers.setdefault("Authorization", f"Basic {b64}")
    elif atype == "hmac":
        secret_env = auth.get("secret_env")
        if secret_env:
            secret = os.getenv(secret_env, "")
            if secret:
                algo = (auth.get("algo") or "sha256").lower()
                header_name = auth.get("header", "X-Signature")
                payload = req.body or b""
                digest = hmac.new(
                    secret.encode(), payload, getattr(hashlib, algo)
                ).digest()
                import base64 as _b64

                req.headers.setdefault(
                    header_name, _b64.b64encode(digest).decode("ascii")
                )
