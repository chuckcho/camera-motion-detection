#!/usr/bin/env bash

VIDEODIR=/media/TB/Videos/Vice
#VIDEOIDS=(02 07 16 18)

python ./cam_detect.py ${VIDEODIR}/VICE_02.mov ./VICE_02.txt VICE_02_overlay.avi
python ./cam_detect.py ${VIDEODIR}/VICE_07.mov ./VICE_07.txt VICE_07_overlay.avi
python ./cam_detect.py ${VIDEODIR}/VICE_16.mov ./VICE_16.txt VICE_16_overlay.avi
python ./cam_detect.py ${VIDEODIR}/VICE_18.mov ./VICE_18.txt VICE_18_overlay.avi
