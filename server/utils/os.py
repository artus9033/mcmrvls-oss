from logging import Logger
import mimetypes
import os
import re
import subprocess

import cv2

def isCommandAvailable(command: str | list[str]):
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def listWorkingCameraPorts(logger: Logger, maxScannedPortIndex: int = 10) -> list[tuple[int, str]]:
    try:
        workingPorts: None | list[tuple[int, str]] = None

        # UNIX discovery
        if os.name != "nt":
            # most robust of the methods: find specifically Logitech BRIO Cams
            if isCommandAvailable(["which", "v4l2-ctl"]):
                result = subprocess.run(
                    ["v4l2-ctl", "--list-devices"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Use grep to filter the output
                grep_command = subprocess.Popen(
                    ["grep", "-E", "BRIO|A4tech", "-C1"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                output, _ = grep_command.communicate(input=result.stdout + result.stderr)  # somehow ffmpeg outputs to stderr...

                if output is None:
                    logger.error("Error checking via command line for cameras on the system")

                    return []

                output = [line for line in output.splitlines() if len(line.strip()) > 0]

                workingPorts = []

                name = output[0].strip()

                for line in output[1:]:
                    pattern = re.compile(r"/dev/video(\d+)$")

                    matches = pattern.findall(line.strip())

                    for match in matches:
                        index = match
                        logger.info(f"> Hit - detected camera on port {index} with v4l2-ctl: {line}")
                        workingPorts.append((int(index), name))

                return workingPorts

            # fallback if v4l2-utils are not installed
            if isCommandAvailable(["which", "ffmpeg"]) and isCommandAvailable(["which", "awk"]):
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-f",
                        "avfoundation",
                        "-list_devices",
                        "true",
                        "-i",
                        '""',
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Use awk to filter the output
                awk_command = subprocess.Popen(
                    [
                        "awk",
                        "/AVFoundation video devices/,/AVFoundation audio devices/",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                output, _ = awk_command.communicate(input=result.stdout + result.stderr)  # somehow ffmpeg outputs to stderr...

                if output is None:
                    logger.error("Error checking via command line for cameras on the system")

                    return []

                # strip the leading & trailing section markers
                output = output.splitlines()[1:-1]

                workingPorts = []

                for line in output:
                    pattern = re.compile(r"\[(\d+)\]\s+(.+)$")

                    matches = pattern.findall(line)

                    for match in matches:
                        index, name = match

                        if "Capture screen" in name or "FaceTime" in name:
                            logger.info(f"Discarding detected camera: Index: {index}, Name: {name}")
                            continue

                        workingPorts.append((int(index), name))

                return workingPorts

        # fallback when ffmpeg / awk are not available & when they fail
        if workingPorts is None:
            workingPorts = []

            portIndex = 0
            while portIndex <= maxScannedPortIndex:
                try:
                    logger.debug(f"Checking port {portIndex}...")

                    camera = cv2.VideoCapture(portIndex)

                    if not camera.isOpened():
                        logger.warning(f"Camera on port {portIndex} cannot be opened")
                    else:
                        is_reading, img = camera.read()
                        w = int(camera.get(3))
                        h = int(camera.get(4))

                        if is_reading:
                            logger.info(f"Camera on port {portIndex} works and has resolution ({h} x {w})")
                            workingPorts.append((portIndex, f"Camera {portIndex} ({h} x {w})"))
                        else:
                            logger.warning(f"Camera on port {portIndex} has resolution {h} x {w} but does not emit data")
                except Exception as e:
                    logger.warning(f"Error testing port {portIndex}: {e}")

                portIndex += 1

            return workingPorts
    except Exception as e:
        logger.error(f"Error listing working camera ports: {e}")
        return []


def isVideoFile(filename: str) -> bool:
    """Checks if the file extension is a mime of video file."""
    mime_type, _ = mimetypes.guess_type(filename)

    return mime_type is not None and mime_type.startswith("video/")
