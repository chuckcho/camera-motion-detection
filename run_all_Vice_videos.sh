#!/usr/bin/env bash

VIDEOS=/media/TB/Videos/Vice/*.mov
for f in ${VIDEOS}; do
  echo Processing video=$f
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  python ./cam_detect.py $f ./${filename}.txt ./${filename}_overlay.avi
done
