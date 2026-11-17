"""Read trusted host credentials without exposing their encoding or contents."""

from __future__ import annotations

import json
from pathlib import Path


def read_trusted_aksk(path: str) -> str:
    """Accept plain AK:SK or the supported JSON formats; sanitize every failure."""
    try:
        raw = Path(path).read_text().strip()
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = raw
        if isinstance(parsed, dict):
            key = parsed.get('aksk') or parsed.get('api_key')
            if not key:
                parts = [parsed.get('access_key_id'), parsed.get('secret_key')]
                if not all(isinstance(part, str) and part and ':' not in part for part in parts):
                    raise ValueError('invalid credential')
                key = ':'.join(parts)
        else:
            key = parsed
        if not isinstance(key, str) or key.count(':') != 1 or any(c.isspace() for c in key) or not all(key.split(':')):
            raise ValueError('invalid credential')
    except Exception:
        raise ValueError('DBox requires a readable trusted credential file containing complete AK:SK') from None
    return key
