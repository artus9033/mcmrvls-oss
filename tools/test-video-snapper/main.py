import datetime
import logging
import time

import cv2
import numpy as np
import rtsp

from snapperutils import *

logger = logging.getLogger("snapper")

frameByFrameMode = False
RTSPs = ["rtsp://mcmrvls.local:8554/unicast", "rtsp://mcmrvls.local:8555/unicast"]


class CVWrapper:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class RTSPWrapper:
    def __init__(self, logger: logging.Logger, rtspSource: str):
        self.logger = logger
        self.rtspSource = rtspSource

        self.logger.info(f"Opening RTSP {rtspSource}")

        self.client = rtsp.Client(rtsp_server_uri=rtspSource, verbose=True)

        # it always fails at the beginning, so we wait until it works
        while not self.client.read():
            print("Waiting for RTSP stream...")
            time.sleep(1)

    def read(self):
        frame = self.client.read()

        if not frame:
            self.logger.error(
                f"Error: Unable to capture image from RTSP {self.rtspSource}"
            )
            return False, None

        return True, cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

    def release(self):
        self.client.close()


CONTROL_HELP = (
    "remember that all inputs are captured to the GUI windows, not the terminal!"
)


def main():
    camera_indices = listWorkingCameraPorts(logger, maxScannedPortIndex=10)
    if not camera_indices:
        print("No cameras found.")
        return

    caps = []
    writers = []
    window_names = []
    frame_counts = [0] * (
        len(camera_indices) + len(RTSPs)
    )  # track frame count per camera

    # Video settings
    fps = 60

    dirname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(dirname, exist_ok=True)

    # Open cameras and create VideoWriters
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            print(f"Camera {idx} could not be opened.")
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"Camera {idx} frame read failed.")
            cap.release()
            continue

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            f"{dirname}/camera_{idx}.avi", fourcc, fps, (frame.shape[1], frame.shape[0])
        )

        wrapper = CVWrapper(cap)

        caps.append(wrapper)
        writers.append(out)
        window_name = f"Camera {idx}"
        window_names.append(window_name)
        cv2.namedWindow(window_name)

    # Open cameras and create VideoWriters
    for source in RTSPs:
        wrapper = RTSPWrapper(logger, source)

        wrapper.read()  # Wait for first frame to get dimensions
        ret, frame = wrapper.read()
        if not ret:
            print(f"RTSP {source} frame read failed.")
            wrapper.release()
            continue

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            f"{dirname}/camera_{source.replace("/", "_").replace(":", "_")}.avi",
            fourcc,
            fps,
            (frame.shape[1], frame.shape[0]),
        )

        caps.append(wrapper)
        writers.append(out)
        window_name = f"RTSP {source}"
        window_names.append(window_name)
        cv2.namedWindow(window_name)

    if not caps:
        print("No cameras could be initialized.")
        return

    paused = False
    start_time = time.time()
    last_print = 0

    print(f"Press 'p' to play/pause, 'q' to quit ({CONTROL_HELP}).")

    while True:
        if not paused:
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    print(f"Camera #{i} frame read failed, stopping...")
                    paused = True
                    break
                writers[i].write(frame)
                frames.append(frame)
                frame_counts[i] += 1

            current_time = time.time() - start_time

            # Print timestamp and frame counts once per second
            if int(current_time) != last_print:
                print(f"Recording timestamp: {current_time:.2f} seconds")
                for i, count in enumerate(frame_counts):
                    print(f"  Camera #{i} frame: {count}")
                last_print = int(current_time)

            # Show frames
            for i, frame in enumerate(frames):
                cv2.imshow(window_names[i], frame)

        if frameByFrameMode and not paused:
            paused = True
            print(
                f"Paused (frame-by-frame mode), press 'p' to record next frame ({CONTROL_HELP})"
            )

        key = cv2.waitKey(30) & 0xFF
        if key == ord("p"):  # toggle pause/play
            paused = not paused
            if paused:
                print("Paused")
            else:
                print("Playing")
        elif key == ord("q"):
            print("Exiting...")
            break

    # Release everything
    for cap in caps:
        cap.release()
    for writer in writers:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
