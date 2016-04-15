#!/usr/bin/env bash

#VIDEODIR=/media/TB/Videos/Vice
#VIDEOIDS=(02 07 16 18)

python ./cam_detect.py /media/TB/Videos/Vice/VICE_02.mov ./VICE_02.txt
python ./cam_detect.py /media/TB/Videos/Vice/VICE_07.mov ./VICE_07.txt
python ./cam_detect.py /media/TB/Videos/Vice/VICE_16.mov ./VICE_16.txt
python ./cam_detect.py /media/TB/Videos/Vice/VICE_18.mov ./VICE_18.txt
