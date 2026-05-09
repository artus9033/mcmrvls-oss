import enum
import os
import re
import threading

import cv2
import numpy as np
import psutil

TRACY_MAX_IMAGE_SIZE = 400  # as per Tracy requirements

emojiRegex = re.compile(r"[\U0001F600-\U0001F64F]")


def stripUTFEmojis(text: str) -> str:
    return emojiRegex.sub("", text)


class ErrorPropagatingTracyZoneWrapper:
    """A wrapper for ScopedZone that does not swallow exceptions raised by the inner block."""

    def __init__(self, *args, **kwargs):
        self.zone = Tracing.tracy_client.ScopedZone(*args, **kwargs)

    def __enter__(self):
        return self.zone.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.zone.__exit__(exc_type, exc_value, traceback)
        if exc_type is not None:
            return False  # propagate exception

    def color(self, *args, **kwargs):
        """Proxy method for color setting."""
        return self.zone.color(*args, **kwargs)

    def text(self, *args, **kwargs):
        """Proxy method for text setting."""
        return self.zone.text(*args, **kwargs)

    def message(self, *args, **kwargs):
        """Proxy method for message logging."""
        return self.zone.message(  # pyright: ignore[reportAttributeAccessIssue]
            *args, **kwargs
        )


class StubScopedZone:
    """A stub for ScopedZone when profiling is not enabled."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False  # propagate exception

    def color(self, *args, **kwargs):
        """Stub method for color setting."""
        pass

    def text(self, *args, **kwargs):
        """Stub method for text setting."""
        pass

    def message(self, *args, **kwargs):
        """Stub method for message logging."""
        pass


class Tracing:
    profiling: bool = False

    _configuredCounters: dict[str, int] = {}

    # colors for traces
    COLOR_GREEN = 0x00FF00
    COLOR_DARKGREEN = 0x005000
    COLOR_ORANGE = 0xFFA500
    COLOR_YELLOW = 0xFFFF00
    COLOR_RED = 0xFF0000
    COLOR_BLUE = 0x0000FF

    # re-export type for ease of use
    class PlotType(enum.Enum):
        """Plot type for plotting numeric values."""

        Number = enum.auto()
        Percentage = enum.auto()

    _memoryThread: threading.Thread | None = None
    _stopMemoryThreadEvent = threading.Event()

    @staticmethod
    def initialize(profiling: bool):
        if not profiling:
            return

        import tracy_client

        Tracing.tracy_client = tracy_client

        Tracing.tracy_client.program_name("MCMRVLS")

        Tracing.profiling = profiling

        Tracing._memoryThread = threading.Thread(
            target=Tracing._memoryReporterThreadWorker,
            name="TracyMemoryReporter",
        )
        Tracing._memoryThread.start()

    @staticmethod
    def _memoryReporterThreadWorker():
        Tracing.threadName("TracyMemoryReporter")
        process = psutil.Process(os.getpid())

        while not Tracing._stopMemoryThreadEvent.is_set():
            if Tracing.isProfiling():
                memBytes = process.memory_info().rss
                Tracing.plot(
                    "Memory (RSS) [MiB]",
                    round(memBytes / (1024 * 1024)),
                    Tracing.PlotType.Number,
                )
            Tracing._stopMemoryThreadEvent.wait(1)

    @staticmethod
    def isProfiling() -> bool:
        return Tracing.profiling

    @staticmethod
    def frameImage(image: np.ndarray):
        if Tracing.isProfiling():
            # as per Tracy requirements: ensure the image is resized so that a single dimension is at most 400px, keeping aspect ratio
            height, width = image.shape[:2]
            if height > width:
                new_height = TRACY_MAX_IMAGE_SIZE
                new_width = int(width * (TRACY_MAX_IMAGE_SIZE / height))
            else:
                new_width = TRACY_MAX_IMAGE_SIZE
                new_height = int(height * (TRACY_MAX_IMAGE_SIZE / width))

            # as per Tracy requirements: ensure both dimensions are divisible by 4
            if new_width % 4 != 0:
                new_width -= new_width % 4
            if new_height % 4 != 0:
                new_height -= new_height % 4

            # the image is BGR, convert it to RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            image[:, :, 3] = 255

            image = cv2.resize(image, (new_width, new_height))

            return Tracing.tracy_client.frame_image(image.tobytes(), new_width, new_height)
        else:
            return False

    @staticmethod
    def threadName(name: str):
        if Tracing.isProfiling():
            Tracing.tracy_client.thread_name(name)

    @staticmethod
    def frameMarkStart(name: str):
        if Tracing.isProfiling():
            return Tracing.tracy_client.frame_mark_start(name)
        else:
            return 0

    @staticmethod
    def frameMarkEnd(id: int):
        if Tracing.isProfiling():
            return Tracing.tracy_client.frame_mark_end(id)
        else:
            return False

    @staticmethod
    def frameMark():
        if Tracing.isProfiling():
            Tracing.tracy_client.frame_mark()

    @staticmethod
    def ScopedZone(name: str):
        if Tracing.isProfiling():
            return ErrorPropagatingTracyZoneWrapper(name)
        else:
            return StubScopedZone(name)

    @staticmethod
    def message(message: str):
        if Tracing.isProfiling():
            return Tracing.tracy_client.message(stripUTFEmojis(message))
        else:
            return False

    @staticmethod
    def plot(counterName: str, value: float | int, plotType: PlotType):
        if Tracing.isProfiling():
            plotId = Tracing._configuredCounters.get(counterName, None)
            if plotId is None:
                plotId = Tracing.tracy_client.plot_config(counterName, Tracing._internalPlotTypeToTracyPlotType(plotType))

                if plotId is None:
                    print(f"Failed to configure plot '{counterName}' with type '{plotType}'.")
                    return

                Tracing._configuredCounters[counterName] = plotId

            return Tracing.tracy_client.plot(plotId, value)
        else:
            return False

    @staticmethod
    def _internalPlotTypeToTracyPlotType(
        plotType: PlotType,
    ) -> int:
        if Tracing.tracy_client is None:
            return 1

        match plotType:
            case Tracing.PlotType.Number:
                return Tracing.tracy_client.PlotFormatType.Number
            case Tracing.PlotType.Percentage:
                return Tracing.tracy_client.PlotFormatType.Percentage
            case _:
                raise ValueError(f"Unknown plot type: {plotType}")

    @staticmethod
    def shutdown():
        Tracing._stopMemoryThreadEvent.set()

        if Tracing._memoryThread is not None:
            Tracing._memoryThread.join()
