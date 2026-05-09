import logging
import os

from utils.constants import serverRootPath
from utils.os import isVideoFile

from ..VideoCaptureWrapper import (
    AbstractVideoCaptureWrapper,
    PlaybackVideoCaptureWrapper,
)
from ..VideoCaptureWrapperFactory import AbstractVideoCaptureWrapperFactory

class PlaybackVideoCaptureWrapperFactory(AbstractVideoCaptureWrapperFactory):
    @staticmethod
    def getPlaybackDirPath(path: str) -> str:
        return os.path.join(serverRootPath, "res", "video", path)

    @staticmethod
    def createVideoCaptureWrappers(logger: logging.Logger, videosDirectoryName: str, loop: bool) -> list[AbstractVideoCaptureWrapper]:
        wrappers: list[AbstractVideoCaptureWrapper] = []

        playbackDir = PlaybackVideoCaptureWrapperFactory.getPlaybackDirPath(videosDirectoryName)

        if not os.path.exists(playbackDir):
            msg = f"Playback directory {playbackDir} does not exist"
            logger.error(msg)
            raise Exception(msg)

        files = os.listdir(playbackDir)

        if len(files) == 0:
            msg = f"No files found in playback directory {playbackDir}"
            logger.error(msg)
            raise Exception(msg)

        for i, file in enumerate(files):
            # check if this is a video file
            if isVideoFile(file):
                wrappers.append(
                    PlaybackVideoCaptureWrapper(
                        logger=logger,
                        video_path=os.path.join(playbackDir, file),
                        capture_index=i,
                        loop=loop,
                    )
                )

        return wrappers
