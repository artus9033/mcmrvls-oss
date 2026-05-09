class SioRoomsStateTracker:
    SIO_ROOM_ALGORITHM_PREVIEWS = "algorithm_previews"
    SIO_ROOM_STITCHING_PREVIEWS = "stitching_previews"
    SIO_ROOM_CAMERA_PREVIEWS = "camera_previews"
    SIO_ROOM_MAP_SEGMENTATION_PREVIEWS = "map_segmentation_previews"
    SIO_ROOM_ALL_DETECTIONS = "all_detections"
    SIO_ROOM_MAP_NUMPY = "map_numpy"
    SIO_ROOM_CACHE_EVENTS = "cache_events"

    _connectedSids: set[int] = set()

    _sioClientsInRooms: dict[str, set[int]] = {
        SIO_ROOM_ALGORITHM_PREVIEWS: set(),
        SIO_ROOM_STITCHING_PREVIEWS: set(),
        SIO_ROOM_MAP_SEGMENTATION_PREVIEWS: set(),
        SIO_ROOM_ALL_DETECTIONS: set(),
        SIO_ROOM_MAP_NUMPY: set(),
        SIO_ROOM_CAMERA_PREVIEWS: set(),
        SIO_ROOM_CACHE_EVENTS: set(),
    }

    @staticmethod
    def handleConnected(sid: int):
        SioRoomsStateTracker._connectedSids.add(sid)

    @staticmethod
    def handleDisconnected(sid: int):
        SioRoomsStateTracker._connectedSids.remove(sid)

    @staticmethod
    def handleRoomJoinedBy(sid: int, room: str):
        SioRoomsStateTracker._sioClientsInRooms[room].add(sid)

    @staticmethod
    def handleRoomLeftBy(sid: int, room: str):
        SioRoomsStateTracker._sioClientsInRooms[room].remove(sid)

    @staticmethod
    def getTotalConnectedCount() -> int:
        return len(SioRoomsStateTracker._connectedSids)

    @staticmethod
    def getClientsCountInRoom(room: str) -> int:
        return len(SioRoomsStateTracker._sioClientsInRooms[room])
