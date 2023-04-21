from typing import Optional, List

from steamship import Tag
from steamship.data.block import Block

def get_tag_value_key(tags: Optional[List[Tag]], kind: str, name: str, key: str) -> Optional[any]:
    """Iterates through a list of tags and returns the first that contains the provided Kind/Name/ValueKey."""
    for tag in tags or []:
        if tag.kind == kind and tag.name == name:
            return (tag.value or {}).get(key)
    return None
