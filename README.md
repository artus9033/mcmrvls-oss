# Multi-Camera Multi-Robot Visual Localization System (M-C M-R VLS)

- [Multi-Camera Multi-Robot Visual Localization System (M-C M-R VLS)](#multi-camera-multi-robot-visual-localization-system-m-c-m-r-vls)
  - [Setup \& running (containerized)](#setup--running-containerized)
    - [Prerequisites](#prerequisites)
    - [All-in-one development compose](#all-in-one-development-compose)
    - [All-in-one production compose](#all-in-one-production-compose)
    - [Developing (containerized)](#developing-containerized)
      - [Server](#server)
      - [Frontend](#frontend)
      - [ROS client package](#ros-client-package)
  - [Setup \& running (non-containerized)](#setup--running-non-containerized)
    - [Server](#server-1)
      - [Prerequisites:](#prerequisites-1)
      - [Setup](#setup)
      - [Running](#running)
    - [React web frontend](#react-web-frontend)
      - [Prerequisites:](#prerequisites-2)
      - [Setup](#setup-1)
      - [Running](#running-1)
      - [Building](#building)
    - [DuckieTown container](#duckietown-container)
      - [Prerequisites](#prerequisites-3)
      - [Running](#running-2)
      - [RVIZ](#rviz)
        - [`roslaunch multicamera_localization robot_localization.launch` options](#roslaunch-multicamera_localization-robot_localizationlaunch-options)
        - [`roslaunch multicamera_localization map_segmentation.launch` options](#roslaunch-multicamera_localization-map_segmentationlaunch-options)
        - [`roslaunch multicamera_localization mcmrvls_full.launch` options](#roslaunch-multicamera_localization-mcmrvls_fulllaunch-options)
      - [Using the ROS dev container](#using-the-ros-dev-container)
  - [April tags used](#april-tags-used)
  - [Calibration procedure](#calibration-procedure)

---

## Setup & running (containerized)

### Prerequisites

- Docker
- Docker compose
- [Just](https://github.com/casey/just) command runner

### All-in-one development compose

Run everything with `docker-compose` via `just`: `just compose-dev-up <duckiebot name>`.
To stop everything, run `just compose-dev-down <duckiebot name>`.
To prune all containers (e.g. before rebuilding & re-starting or for cleanup), run `just compose-prune`.
To run everything & force a rebuild of images, run `just compose-dev-up-rebuild <duckiebot name>`.
To run down, prune & then up with rebuild at once, run `compose-dev-clean-recreate`.

> ![INFO]
> Above, `<duckiebot name>` means the robot name without `.local` - e.g. `d4`.

### All-in-one production compose

The production compose contains the server & frontend. The ROS package in production shall be run on separate machine(s), thus is not included.

Run everything with `docker-compose` via `just`: `just compose-prod-up`.
To stop everything, run `just compose-prod-down`.
To prune all containers (e.g. before rebuilding & re-starting or for cleanup), run `just compose-prune`.
To run everything & force a rebuild of images, run `just compose-prod-up-rebuild`.
To run down, prune & then up with rebuild at once, run `compose-dev-clean-recreate`.

### Developing (containerized)

#### Server

1. Attach VS Code to the container (open folder: `/server/`)
2. Run the server, e.g.:
   - `just live` for running live
   - `just video <filename>` for running from playback, where `<filename>` is the directory name inside `/server/res/video/` that contains files named `<0..n>.avi` that are camera feeds and must be of equal length

#### Frontend

1. Attach VS Code to the container (open folder: `/frontend/`)
2. `pnpm start`

#### ROS client package

1. Attach VS Code to the container (open folder: `/code/catkin_ws/src/multicamera_localization`)
2. ~~`source /opt/ros/noetic/setup.bash`~~ can be skipped since `~/.bashrc` already has this command appended
3. `just rebuild`
4. `source devel/setup.bash`
5. `just setup`
6. `just containerized-start`

---

## Setup & running (non-containerized)

### Server

#### Prerequisites:

1. Python >= 3.12

#### Setup

1. Create a venv & enable it: `python -m venv ./venv`
2. Build & install Tracy profiler (requires interaction!):
   - `cd server`
   - `./installTracyPython.sh` on Linux / MacOS
   - `./installTracyPython.bat` on Windows
3. Install the just command runner: `cargo install just`
4. `cd server`
5. `pip install -r requirements.txt`

#### Running

From `server/`:

- in still mode from a video file: `just video <name>` where `<name>` is the directory name inside `server/res/video/` that contains files named `<0..n>.avi` that are camera feeds and must be of equal length; if the directory also contains `fps.txt`, the value inside will be used as the playback fps, otherwise there will be no delay between frames
- live from plugged in USB cameras: `just live`

### React web frontend

#### Prerequisites:

1. Node.JS >= 21 with corepack

#### Setup

1. `cd frontend`
2. `corepack enable`
3. `corepack install`
4. `pnpm`

#### Running

From `frontend/`: `pnpm start`

#### Building

To build a production bundle, run `pnpm build` from `frontend/`. Outputs will be written to `frontend/build`.

### DuckieTown container

#### Prerequisites

- Docker
- DTS (DuckieTown Shell)

#### Running

Build the DuckieTown container:

```bash
cd duckietown
just build
```

And launch it either locally binding to the ROS master of the given robot `<ROBOT_NAME>.local`:

```bash
# d2  -> the robot's hostname without .local suffix
# 406 -> the robot's April Tag ID
just run-driver d2 406
```

Or publish on the remote Docker onboard the robot `<ROBOT_NAME>.local`:

```bash
dts devel run -A "406" -H ROBOT_NAME
```

Reminder of how DT works: the entrypoint (launch file + args) is `duckietown/launchers/default.sh`.

#### RVIZ

Terminal A: `cd ./duckietown`
Terminal B: `cd ~/DTF/scripts`
Terminal C: `cd ./server`
Terminal D: `cd ./frontend`

```
# terminal C - launch the M-C M-R VLS backend
just live

# terminal D - launch the M-C M-R VLS frontend
yarn start

# terminal B
./container_gui_tools.sh d2

# terminal A
docker cp packages/robot_driver/rviz/driving.rviz gui_container_name:/home/duckie/
just run-driver d2 406

# terminal B
rviz -d /home/duckie/driving.rviz
```

---

Common options for **all launch files** include:

- `server_address`: address of the M-C M-R VLS server (default: `localhost`)
- `server_port`: port of the M-C M-R VLS server (default: `6001`)

##### `roslaunch multicamera_localization robot_localization.launch` options

Common options and:

- `robot_id`: ID of the robot (**no default value, mandatory to be passed**)

##### `roslaunch multicamera_localization map_segmentation.launch` options

Only common options apply.

##### `roslaunch multicamera_localization mcmrvls_full.launch` options

Options same as for `robot_localization.launch`.

#### Using the ROS dev container

To use the ROS dev container, keep in mind that M-C M-R VLS container has the `server` hostname. For instance, you can run:

```bash
cd /code/catkin_ws
catkin build
source devel/setup.bash
cd src/multicamera_localization
roslaunch robot_driver duckietown.launch robot_id:=580
```

## April tags used

For all four corners of the map + N inner points, the highest N + 4 IDs of `tag36h11` are used:

- `tag36_11_00579`
- `tag36_11_00580`
- `tag36_11_00581`
- `tag36_11_00582`
- `tag36_11_00583`
- `tag36_11_00584`
- `tag36_11_00585`
- `tag36_11_00586`

## Calibration procedure

1. Calibrate camera using chessboard
2. Calculate homography for transforming images to a common perspecitve
