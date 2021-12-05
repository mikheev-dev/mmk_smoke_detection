from typing import Any, Optional, Callable

import os


def get_env(
        name: str,
        default: Optional[Any] = None,
        cast: Optional[Callable] = None
) -> Any:
    env = os.environ.get(name, default=default)
    if not cast:
        return env
    return cast(env)
