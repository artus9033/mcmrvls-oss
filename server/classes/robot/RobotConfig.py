from .RobotKalmanParams import RobotKalmanParams


class RobotConfig:
    id: int
    host: str
    kalman_params: RobotKalmanParams

    def __init__(self, id: int, host: str, kalman_params: RobotKalmanParams | None = None) -> None:
        self.id = id
        self.host = host
        self.kalman_params = kalman_params if kalman_params is not None else RobotKalmanParams.hard_defaults()

    def toDTO(self):
        return {
            "id": self.id,
            "host": self.host,
        }

    def __key__(self) -> int:
        return self.id

    def __hash__(self) -> int:
        return self.__key__()

    def __eq__(self, value: object) -> bool:
        if isinstance(value, int):
            return self.__key__() == value
        else:
            return self is value

    def __str__(self) -> str:
        return f"RobotConfig(id={self.id}, host={self.host})"

    def __repr__(self) -> str:
        return str(self)
