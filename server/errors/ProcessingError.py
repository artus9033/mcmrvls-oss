from typing import Any

class ProcessingError(Exception):
    fullExplanation: str
    code: str
    value: Any

    def __init__(self, fullExplanation: str, code: str, value: Any = None) -> None:
        super().__init__(fullExplanation)
        self.fullExplanation = fullExplanation
        self.code = code
        self.value = value

    def __str__(self) -> str:
        return self.fullExplanation

    def __repr__(self) -> str:
        return str(self)
