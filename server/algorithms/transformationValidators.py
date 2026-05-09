def validateAspectRatioClose(targetAspect: float, actualAspect: float, tolerancePercent: float) -> bool:
    """Check if an aspect ratio is achieved with a given tolerance."""
    lower_bound = targetAspect * (1 - tolerancePercent / 100)
    upper_bound = targetAspect * (1 + tolerancePercent / 100)

    return lower_bound <= actualAspect <= upper_bound
