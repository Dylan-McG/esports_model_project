from __future__ import annotations

import os
from typing import Optional


def get_opendota_api_key() -> Optional[str]:
    # 1) Try OS keychain
    try:
        import keyring

        key = keyring.get_password("esports_quant", "opendota_api_key")
        if key:
            return key
    except Exception:
        pass

    # 2) Try .env fallback
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    return os.getenv("OPENDOTA_API_KEY")
