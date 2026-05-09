from abc import ABC

from utils.constants import disposeStitchingResourceHoldersAfterUnusedRounds

class RoundCacheResource(ABC):
    """Base class for a resource that may or may not be needed anymore after it is unused after a specific number of rounds."""

    def __init__(self) -> None:
        self.idleRoundsInTurn = -1  # -1 as it will be soon incremented after usage mark - at the end of round

    def markRoundAndCheckIfNeeded(self) -> bool:
        """Mark a new round of stitching; returns True if it is still needed, False if it can be disposed."""
        self.idleRoundsInTurn += 1

        return self.idleRoundsInTurn < disposeStitchingResourceHoldersAfterUnusedRounds

    def markUsage(self):
        """Mark that the resource has just been used and reset idle counter."""
        self.idleRoundsInTurn = -1  # -1 as it will be soon incremented after usage mark - at the end of round

    def wasActiveLastRound(self) -> bool:
        """Check if the resource was active in the last round."""
        return self.idleRoundsInTurn <= 0
