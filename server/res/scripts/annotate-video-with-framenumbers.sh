ffmpeg -i 1.avi \
   -qscale 0 -vf "drawtext=fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099" \
    1-ann.avi