from functools import total_ordering
from typing import Self

@total_ordering
class StitchingStageDescription:
    def __init__(
        self,
        previous: Self | str,
        next: Self | str | None = None,
    ):
        self.previous = previous
        self.next = next

    def cloneWithNext(self, next: str) -> "StitchingStageDescription":
        return StitchingStageDescription(self, next)

    def includes(self, stage: Self) -> bool:
        if self.next is None:
            return stage == self.previous
        else:
            nextHasStage = False
            if isinstance(self.next, StitchingStageDescription):
                nextHasStage = self.next.includes(stage)
            else:
                nextHasStage = self.next == stage.__str__()
            return stage == self.previous or nextHasStage

    def __contains__(self, stage: Self) -> bool:
        return self.includes(stage)

    def __str__(self):
        if self.next is None:
            return f"{self.previous.__str__()}"
        else:
            return f"({self.previous.__str__()}+{self.next.__str__()})"

    def __repr__(self):
        return f"StitchingStageDescription<{self.__str__()}>"

    def __eq__(self, other):
        if isinstance(other, StitchingStageDescription):
            return self.previous == other.previous and self.next == other.next
        elif isinstance(other, str):
            return self.__str__() == other
        else:
            return False

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, StitchingStageDescription):
            return NotImplemented

        return self.__str__() < other.__str__()

    def __hash__(self):
        return hash((self.previous.__str__(), self.next.__str__()))
