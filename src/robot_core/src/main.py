#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import math

import rospy
from geometry_msgs.msg import Point, Twist

import cv2 as cv
from pupil_apriltags import Detector

pubVel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
pubPoint = rospy.Publisher('apriltag_location', Point, queue_size=10)
rospy.init_node('apriltag_finder', anonymous=True)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0

    while True:
        start_time = time.time()

        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        debug_image = draw_tags(debug_image, tags, elapsed_time)

        elapsed_time = time.time() - start_time

        key = cv.waitKey(1)
        if key == 27:
            break

        cv.imshow('AprilTag Detect Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        #print(center)
        corners = tag.corners
        #Use corners to detetermine size later
        point_message = Point()
        point_message.x = center[0] #x and y is in pixels
        point_message.y = center[1]
        
        
        

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 =(int(corners[3][0]), int(corners[3][1])) 
        Vx = corner_04[0]- corner_01[0]
        Vy =corner_04[1]- corner_01[1]
        Ux = corner_02[0]- corner_01[0]
        Uy =corner_02[1]- corner_01[1]
        Ameas = Vx*Uy-Vy*Ux
        dist = 1000**0.5 * pow(Ameas, -0.5)

        point_message.z = dist #z is in meters 
        #rospy.loginfo(point_message)
        #pubPoint.publish(point_message)
        vel_message = Twist()
        if(center[0] >= 400):
            vel_message.angular.z = -0.5
        elif(center[0]<= 240):
            vel_message.angular.z = 0.5
        elif(center[0]> 240 and center[0] < 400):
            vel_message.linear.x = 1

        pubVel.publish(vel_message)
        #rospy.loginfo(vel_message)

        x = math.sin((point_message.x - 320) / 640 * math.pi / 6) * point_message.z
        point_message.x = x

        rospy.loginfo(point_message)
        pubPoint.publish(point_message)

        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image

if __name__ == '__main__':
    main()
