from typing import List, Dict, Optional


def get_messages_list(
    user: str,
    system: Optional[str] = None
) -> List[Dict[str, str]]:

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages
