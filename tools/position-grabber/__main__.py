import datetime
import os
import time
import traceback

import pandas as pd
import socketio

SIO_SERVER_URL = "ws://localhost:6001"
print(f"Connecting to {SIO_SERVER_URL}...")

while True:
    try:
        client = socketio.SimpleClient()
        client.connect(url=SIO_SERVER_URL)
        break
    except Exception as e:
        print(f"Connection failed with exception {e}, retrying in 2s...")
        time.sleep(1)

client.emit("subscribe_all_detections")

buffer = pd.DataFrame(
    columns=["time", "robotID", "x", "y", "theta", "raw_x", "raw_y", "raw_azimuth"]
)

startTime = datetime.datetime.now().strftime("%d.%m.%Y-%H_%M_%S")
timestampStart = time.time()


def finalizeData():
    outDir = os.path.realpath(os.path.join(os.path.dirname(__file__), "out"))
    outPath = os.path.join(outDir, f"data-{startTime}.csv")

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    buffer.to_csv(outPath)


print("Connected. Capturing data, press Ctrl+C to stop")

try:
    try:
        while True:
            try:
                [title, *args] = client.receive(timeout=0.5)

                if title == "all_detections":
                    dct = args[0]

                    for detObj in dct["detections"]:
                        robot = detObj["robot"]
                        robot_id = robot["id"] if isinstance(robot, dict) else robot
                        buffer.loc[len(buffer)] = {
                            "time": time.time() - timestampStart,
                            "robotID": robot_id,
                            "x": detObj["x"],
                            "y": detObj["y"],
                            "theta": detObj["azimuth"],
                            "raw_x": detObj.get("raw_x"),
                            "raw_y": detObj.get("raw_y"),
                            "raw_azimuth": detObj.get("raw_azimuth"),
                        }
                        print(".", end="", flush=True)
                else:
                    print(f"Ignoring message '{title}'")
            except socketio.exceptions.TimeoutError:
                pass
    except KeyboardInterrupt:
        print()
        print("Finishing on user request")
        client.disconnect()
        finalizeData()
except Exception as e:
    print("Uncaught exception intercepted, finishing cleanly", e)
    traceback.print_exc()
    finalizeData()
