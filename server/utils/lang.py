import datetime
from typing import Any, Iterable

import numpy as np

def ensureNumericPrimitive(obj: Any) -> Any:
    """If needed, converts any numpy numeric type to a primitive type."""
    if isinstance(obj, int) or isinstance(obj, float):
        return obj

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        raise ValueError(f"ensurePrimitive: unsupported type {type(obj)}")


def ensureNumericPrimitiveTuple(lst: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(ensureNumericPrimitive(item) for item in lst)


def genRunDatetimeStr() -> str:
    """Generates a formatted timestamp from the current time."""
    return datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
