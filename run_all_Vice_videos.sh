#!/usr/bin/env bash

VIDEOS=~/Vice/*.mov
for f in ${VIDEOS}; do
  echo Processing video=$f
  filename=$(basename "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  python ./cam_detect.py $f > ./${filename}.txt
done
