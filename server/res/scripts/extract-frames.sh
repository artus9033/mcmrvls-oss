ffmpeg -i 1.avi \
   -qscale 0 -vf select='eq(n\,17)+eq(n\,27),setpts=N/FRAME_RATE/TB' -r 10 \
    1-sel.avi