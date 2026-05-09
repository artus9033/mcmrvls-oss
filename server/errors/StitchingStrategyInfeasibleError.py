from classes.resources.ImagePack import ImagePack

class StitchingStrategyInfeasibleError(Exception):
    def __init__(
        self,
        pack1: ImagePack | None,
        pack2: ImagePack | None,
        reason: str | None = None,
    ) -> None:
        super().__init__()

        self.pack1 = pack1
        self.pack2 = pack2
        self.reason = reason
