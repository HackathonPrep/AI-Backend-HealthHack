from functools import lru_cache

import jwt
from fastapi import HTTPException, Request, status
from jwt import PyJWKClient

from app.core.config import get_settings


@lru_cache
def _jwks_client(url: str) -> PyJWKClient:
    return PyJWKClient(url)


async def get_current_user_id(request: Request) -> str:
    settings = get_settings()
    authorization = request.headers.get("Authorization", "")
    if not settings.supabase_enabled or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        key = _jwks_client(settings.supabase_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["ES256", "RS256"], audience="authenticated")
        return claims["sub"]
    except (jwt.PyJWTError, KeyError) as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.") from error
