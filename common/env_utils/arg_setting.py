from typing import Optional
import os


def set_hf_token(token: Optional[str] = None) -> None:
    if token is None or token == "":
        # if passed token is None we making sure a valid env var exists
        existing_hf_token = os.environ.get("HF_TOKEN")
        if existing_hf_token is None or existing_hf_token == "":
            raise ValueError(
                "Invalid `hf_token` value (empty string / None) AND " +
                "Could not find a valid env `HF_TOKEN`"
            )
    else:
        os.environ["HF_TOKEN"] = token
