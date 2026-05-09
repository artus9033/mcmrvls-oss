#!/bin/sh

# check if 1 arg is present
if [ "$#" -eq 1 ]; then
  export RUN_FROM_PLAYBACK_DIRECTORY="$1"
fi

cd /server
/usr/bin/env python .
