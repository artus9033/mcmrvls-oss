#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# AGH note:
# 1. robot_id is the 1st positional arg for this launcher script,
# which must be passed to the container when running it,
# and it must be the April Tag ID for this robot, e.g. 406
# 2. VEHICLE_NAME is the env var set by DuckieTown when running this container, e.g. d2.local

# check if at least one argument is given
if [ $# -lt 1 ]; then
  echo "Usage: $0 <robot's April Tag ID>"
  exit 1
fi

robotId=$1

# note: if the first arg was passed via `dts run ... -A <value>`, then dts will prefix it with '--', which needs to be corrected
case "$robotId" in
  --*) robotId=${robotId#--} ;;
esac

echo "Running on vehicle ${VEHICLE_NAME}"
echo "Robot's April Tag ID is $robotId"

dt-exec roslaunch robot_driver duckietown.launch veh:=${VEHICLE_NAME} robot_id:=$robotId


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
