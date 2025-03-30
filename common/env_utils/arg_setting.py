import os


def set_hf_token(token: str) -> None:
    if token is None or token == "":
        raise ValueError("Invalid `hf_token` value (empty string / None)")
    os.environ["HF_TOKEN"] = token
