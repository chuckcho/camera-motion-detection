#!/usr/bin/env python

"""
Scripts for (1) stationery camera detection and (2) night scene detection
"""

import sys
import numpy as np
import math
#from common.dextro_logger import LOGGER

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
    #LOGGER.info('%s | calculate_motion_and_jitterness: Running on a non-CUDA '
    print('[error] calculate_motion_and_jitterness: Running on a non-CUDA '
        'server.')

def find_dominant_mag_ang(flow):
    """
    Find a dominant magnitude and angle given optical flow map
    """
    mag_map, ang_map = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # If mean(mag) >= thresh1 and std(mag) <= thresh2, this magnitude is
    # considered "dominant"
    min_mag_mean = 0.1 * mag_map.shape[0]/50
    # max mag std deviation relative to mean
    max_mag_std = 0.9
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

def detect_pan_tilt_zoom(videofile, OF_overlay_videofile=None):
    """
    Detect Pan/Tilt/Zoom camera motion separately
    """

    # display images for debugging/troubleshooting
    visualize = False
    debug = True
    save_OF_overlay = True

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

    frame_nums = []
    timestamps = []
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
        timestamp = frame_num / fps
        frame_nums.append(frame_num)
        timestamps.append(timestamp)

        if debug:
            print "[debug] f={}, t={}, dom_mag={}, dom_ang={}".format(
                    frame_num,
                    timestamp,
                    dom_mag,
                    dom_ang)

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

    # detect pan/tilt/zoom for each frame from dom_mag and dom_ang's (only
    # if they persists in some consecutive frames)

    pan = [False] * len(frame_nums)
    tilt = [False] * len(frame_nums)
    zoom = [False] * len(frame_nums)

    min_consecutive_frames = 4
    for count, frame in enumerate(frame_nums[:-min_consecutive_frames]):
        this_clip_pan_or_tilt = True

        for i in range(count, count + min_consecutive_frames):
            if math.isnan(cummulative_dom_mag[i]) or \
                    math.isnan(cummulative_dom_ang[i]):
                this_clip_pan_or_tilt = False

        if this_clip_pan_or_tilt:
            if (cummulative_dom_ang[count] >= 45+20 and \
                cummulative_dom_ang[count] <= 135-20) or \
               (cummulative_dom_ang[count] >= 225+20 and \
                cummulative_dom_ang[count] <= 315-20):
                for i in range(count, count + min_consecutive_frames + 1):
                    if not pan[i]:
                        tilt[i] = True
            else:
                for i in range(count, count + min_consecutive_frames + 1):
                    if not tilt[i]:
                        pan[i] = True

    if OF_overlay_videofile:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(OF_overlay_videofile, fourcc, fps/sampling_rate*2, frame1.shape[1::-1])

        # get FPS
        cap = cv2.VideoCapture(videofile)

        # read first frame and resize
        ret, frame1 = cap.read()
        frame1 = cv2.resize(frame1, (0, 0), fx=resize_ratio, fy=resize_ratio)
        previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        frame_num = 1
        count = 0

        while 1:
            # read subsequent frame
            ret, frame2 = cap.read()

            # check for end of video
            if not ret:
                break

            # skip frames
            #if frame_num % int(round(fps/sampling_rate)) != 0:
            if frame_num % sampling_rate != 0:
                frame_num += 1
                continue

            # resize
            frame2 = cv2.resize(frame2, (0, 0), fx=resize_ratio, fy=resize_ratio)
            if OF_overlay_videofile:
                tmp_frame = frame2
                font = cv2.FONT_HERSHEY_SIMPLEX
                (width, height) = frame2.shape[1::-1]
                if tilt[count]:
                    cv2.putText(tmp_frame,'Tilt',(10,100), font, 1,(0,0,255),2,cv2.LINE_AA)
                elif pan[count]:
                    cv2.putText(tmp_frame,'Pan',(10,100), font, 1,(0,255,255),2,cv2.LINE_AA)
                out.write(tmp_frame)

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

            previous_frame = next_frame
            frame_num += 1
            count += 1

        cap.release()

    return (pan,
            tilt,
            zoom,
            frame_nums,
            timestamps,
            cummulative_dom_mag,
            cummulative_dom_ang)

def main():
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        print "Video file must be specified."
        sys.exit(-1)

    if len(sys.argv) > 2:
        outfile = sys.argv[2]
    else:
        outfile = None

    if len(sys.argv) > 3:
        overlay_video = sys.argv[3]
    else:
        overlay_video = None

    (pan, tilt, zoom, \
     frame_nums, timestamps, \
     dom_mag, dom_ang) = detect_pan_tilt_zoom(video,
             OF_overlay_videofile=overlay_video)

    # human-friendly print out: video, pan, tilt, zoom
    print "video=\"{}\", pan={}, tilt={}, zoom={}".format(
            video,
            any(pan),
            any(tilt),
            any(zoom)
            )

    # save frame-by-frame stats
    if outfile:
        f = open(outfile, 'w')
        f.write('frame_num, time in sec, dominant OF mag, dominant OF ang, pan, tilt, zoom\n')
        for count, frame in enumerate(frame_nums[:-2]):
            f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(
                    frame_nums[count],
                    timestamps[count],
                    dom_mag[count],
                    dom_ang[count],
                    pan[count],
                    tilt[count],
                    zoom[count]
                    ))

if __name__ == "__main__":
    main()
