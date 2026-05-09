import time

class Timer:
    def __init__(self, duration_seconds: int, initial_elapsed_state: bool = False):
        """Initialize the timer with a duration in seconds.
        :param duration: Time in seconds after which the timer elapses.
        :param initial_elapsed_state: Whether the timer should start in an elapsed state.
        """
        self.duration = duration_seconds
        self.start_time = None

        if not initial_elapsed_state:
            self.start()

    def start(self):
        """Start / restart the timer."""
        self.start_time = time.time()

    def hasElapsed(self) -> bool:
        """Check if the specified duration has elapsed since start was called.
        :return: True if the duration has elapsed, False otherwise.
        """
        if self.start_time is None:
            # initial elapsed state
            return True

        return self.getTimeSinceStart() >= self.duration

    def getTimeSinceStart(self) -> float:
        """Get the time elapsed since the timer was started.
        :return: Time elapsed since the timer was started (in seconds).
        """
        return time.time() - self.start_time
