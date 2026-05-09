from typing import Iterable

def minmax(lst: Iterable):
    if not lst:
        raise ValueError("minmax() arg is an empty sequence")

    maxVal = lst[0]
    minVal = lst[0]

    for i in lst:
        if i > maxVal:
            maxVal = i
        if i < minVal:
            minVal = i

    return minVal, maxVal
