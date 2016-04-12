#!/usr/bin/env bash

for f in /media/TB/Videos/Vice/*.mov; do
  echo Processing video=$f
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  python ./cam_detect.py $f > ./${filename}.txt
done
