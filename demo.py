#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import argparse
import colorsys
import os
import random
import datetime
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from yolo import YOLO
from collections import deque
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet


def main(yolo, video_route):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    car_counter = []
    car_total = int(0)
    flag = int(0)
    countss = 1

    kk = 0
    xj = []
    yj = []

    hsv_tuples = [(x / 80, 1., 1.)
                  for x in range(80)]
    color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    color = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            color))
    random.seed(10101)
    random.shuffle(color)
    random.seed(None)


    fileNames = os.path.basename(video_route)

    (fileName, extension) = os.path.splitext(fileNames)


    model_filename = 'model_data/mars-second.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(video_route)

    rate = video_capture.get(5)
    print('rate=' + str(rate))
    frameNumber = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frameNumber / rate
    if duration > 60:
        duration /= 60
        flag = 1
        print('视频时长：' + str(duration) + '分钟')
    elif duration > 3600:
        duration = duration / 60 / 60
        flag = 2
        print('视频时长：' + str(duration) + '小时')
    else:
        print('视频时长：' + str(duration) + '秒')
    if writeVideo_flag:

        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./results/' + fileName + '.avi', fourcc, rate, (w, h))

        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        t1 = time.time()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.uint8(image))

        boxs, class_names, socres, image = yolo.detect_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        features = encoder(frame, boxs)

        detections = [Detection(bbox, 1.0, feature, class_namess) for bbox, feature, class_namess in
                      zip(boxs, features, class_names)]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        i = int(0)
        i1 = int(0)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:  # 状态为confirmed或者time_since_update==0的才绘画出来
                continue
            if track.class_namess == ['car']:
                car_counter.append(int(track.track_id))
                color_type = 2  # 颜色标志位

            bbox = track.to_tlbr()

            cv2.putText(image, str(track.track_id), (int(bbox[0] + 65), int(bbox[1]) - 5), 0, 5e-3 * 100,
                        (color[color_type]), 2)

            if track.class_namess == ['car']:
                i1 = i1 + 1

        count1 = len(set(car_counter))
        car_total = count1
        cv2.putText(image, "Total car Counter: " + str(count1), (int(20), int(80)), cv2.FONT_HERSHEY_SIMPLEX,
                     5e-3 * 100, (0, 0, 255), 2)
        cv2.putText(image, "Current car Counter: " + str(i1), (int(20), int(60)), cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 100,
                    (0, 0, 255), 2)
        cv2.putText(image, "FPS: %f" % (fps), (int(20), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 100, (0, 0, 255), 2)

        cv2.namedWindow('result', 0)
        cv2.imshow('result', image)
        if writeVideo_flag:
            out.write(image)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        xj.append(fps)
        yj.append(kk)
        kk = kk + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    car_flow = format(car_total / duration, '.2f')
    if flag == 1:
        print('car车流量:' + str(car_flow) + '分钟')
    elif flag == 2:
        print('car车流量:' + str(car_flow) + '小时')
    else:
        print('car车流量:' + str(car_flow) + '秒')


    f = open('./results/' + fileName + '.txt', "a")

    # 获取当前时间并格式化
    now_time = datetime.datetime.now().strftime('%Y{y}%m{m}%d{d}%H{h}%M{f}%S{s}').format(y='年', m='月', d='日', h='时',
                                                                                         f='分', s='秒')
    print('写入时间:' + now_time)

    accurate_time = format(duration, '.2f')

    if flag == 1:
        f.write('写入时间:' + now_time + '\n'
                + '视频时长:' + str(accurate_time) + '分钟' + '\n'
                + 'car数目:' + str(car_total) + '辆' + '\n'
                + 'car车流量：' + str(car_flow) + '辆/分钟' + '\n'
                + '\n' + '\n' + '\n')
    elif flag == 2:
        f.write('写入时间:' + now_time + '\n'
                + '视频时长:' + str(accurate_time) + '小时' + '\n'
                + 'car数目:' + str(car_total) + '辆' + '\n'
                + 'car车流量：' + str(car_flow) + '辆/小时' + '\n'
                + '\n' + '\n' + '\n')
    else:
        f.write('写入时间:' + now_time + '\n'
                + '视频时长:' + str(accurate_time) + '秒' + '\n'
                + 'car数目:' + str(car_total) + '辆' + '\n'
                + 'car车流量：' + str(car_flow) + '辆/秒' + '\n'
                + '\n' + '\n' + '\n')
    f.close()

    plt.figure()
    plt.plot(yj[1:], xj[1:])
    plt.savefig("easyplot.jpg")
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--input", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args.input)
    main(YOLO(), args.input)
