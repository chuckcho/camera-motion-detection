#!/usr/bin/env python

"""
Scripts for (1) stationery camera detection and (2) night scene detection
"""

import sys
import numpy as np
import math
from common.dextro_logger import LOGGER

try:
    import cv2
    # normalize some property names across opencv versions
    try:
        from cv2 import CAP_PROP_FPS
        from cv2 import CAP_PROP_FRAME_COUNT
    except ImportError:
        from cv2.cv import CV_CAP_PROP_FPS as CAP_PROP_FPS
        from cv2.cv import CV_CAP_PROP_FRAME_COUNT as CAP_PROP_FRAME_COUNT
except ImportError as import_error:
    LOGGER.info('%s | calculate_motion_and_jitterness: Running on a non-CUDA '
        'server.', import_error)

def find_dominant_mag_ang(flow):
    """
    Find a dominant magnitude and angle given optical flow map
    """
    mag_map, ang_map = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # If mean(mag) >= thresh1 and std(mag) <= thresh2, this magnitude is
    # considered "dominant"
    min_mag_mean = 0.3 * mag_map.shape[0]/50
    # max mag std deviation relative to mean
    max_mag_std = 0.7
    mag_mean = np.mean(mag_map)
    mag_std = np.std(mag_map)
    if mag_mean >= min_mag_mean and mag_std <= max_mag_std * mag_mean:
        dom_mag = mag_mean
    else:
        dom_mag = float('nan')

    # If std(ang) <= thresh3, this angle is considered "dominant"
    # Take cos() to wrap inherently circular angle (0~2*pi, 0=2*pi)
    max_ang_std = 0.6
    ang_std = np.std(np.cos(ang_map))
    if ang_std <= max_ang_std:
        dom_ang = np.mean(ang_map) * 180 / np.pi
    else:
        dom_ang = float('nan')

    # Only if both dom_mag and dom_ang are good, this frame is good
    #if math.isnan(dom_mag):
    #    dom_ang = float('nan')
    #if math.isnan(dom_ang):
    #    dom_mag = float('nan')

    return dom_mag, dom_ang

def detect_pan_tilt_zoom(videofile):
    """
    Detect Pan/Tilt/Zoom camera motion separately
    """

    # display images for debugging/troubleshooting
    visualize = False

    # frames per second (skip other frames)
    # process only every n-th frame
    sampling_rate = 5

    # image resize ratio
    resize_ratio = 0.5

    # will ignore short segments of frames in motion (likely to be noisy)
    # min time span = min_frames_for_motion/fps/sampling_rate (second)
    min_frames_for_motion = 2

    # get FPS
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(CAP_PROP_FPS)

    # if unavailable, by default 30.0
    if fps <= 0.0 or math.isnan(fps):
        fps = 30.0

    # read first frame and resize
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (0, 0), fx=resize_ratio, fy=resize_ratio)
    previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    if visualize:
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

    cummulative_dom_mag = []
    cummulative_dom_ang = []
    frame_num = 1
    count = 1

    if visualize:
        plot_window_size = 500
        cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('optical flow', cv2.WINDOW_NORMAL)
        cv2.namedWindow('dominant mag(OF)', cv2.WINDOW_NORMAL)
        cv2.namedWindow('dominant ang(OF)', cv2.WINDOW_NORMAL)

    while 1:
        # read subsequent frame
        ret, frame2 = cap.read()

        # check for end of video
        if not ret:
            if visualize:
                k = cv2.waitKey(0)
            break

        # skip frames
        #if frame_num % int(round(fps/sampling_rate)) != 0:
        if frame_num % sampling_rate != 0:
            frame_num += 1
            continue

        # resize
        frame2 = cv2.resize(frame2, (0, 0), fx=resize_ratio, fy=resize_ratio)
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # get optical flow
        # refer to http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        # for details about each parameter
        flow = cv2.calcOpticalFlowFarneback(
                prev=previous_frame,
                next=next_frame,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
                )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if visualize:
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # find majority angle and magnitute
        dom_mag, dom_ang = find_dominant_mag_ang(flow)

        cummulative_dom_mag.append(dom_mag)
        cummulative_dom_ang.append(dom_ang)

        # if enabled, will display (1) original image, (2) optical flow image,
        # and (3) history of dominant optical flow angles
        if visualize:
            cummulative_dom_mag_img = np.zeros(
                                        (180, plot_window_size, 3),
                                        np.uint8
                                        )
            for i in xrange(max(0, count - plot_window_size), count):
                cv2.circle(
                    cummulative_dom_mag_img,
                    (
                    count-i,
                    cummulative_dom_mag_img.shape[0] - max(
                            int(cummulative_dom_mag[i])*10, 0)
                    ),
                    1, (0, 0, 255), 1)
            cummulative_dom_ang_img = np.zeros(
                                        (180, plot_window_size, 3),
                                        np.uint8
                                        )
            for i in xrange(max(0, count - plot_window_size), count):
                cv2.circle(
                    cummulative_dom_ang_img,
                    (
                    count-i,
                    cummulative_dom_ang_img.shape[0] - max(
                            int(cummulative_dom_ang[i]), 0)
                    ),
                    1, (0, 0, 255), 1)
            cv2.imshow('original', frame2)
            cv2.imshow('optical flow', bgr)
            cv2.imshow('dominant mag(OF)', cummulative_dom_mag_img)
            cv2.imshow('dominant ang(OF)', cummulative_dom_ang_img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        previous_frame = next_frame
        frame_num += 1
        count += 1

    cap.release()

    print("[debug] cummulative_dom_mag={}".format(cummulative_dom_mag))
    print("[debug] cummulative_dom_ang={}".format(cummulative_dom_ang))

    pan_detected = False
    tilt_detected = False
    zoom_detected = False

    return pan_detected, tilt_detected, zoom_detected

def main():
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        print "Video file must be specified."
        sys.exit(-1)

    pan, tilt, zoom = detect_pan_tilt_zoom(video)

    # human-friendly print out: video, pan, tilt, zoom
    print "video=\"{}\", pan={}, tilt={}, zoom={}".format(
            video,
            pan,
            tilt,
            zoom
            )

if __name__ == "__main__":
    main()
