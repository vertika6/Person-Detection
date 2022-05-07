import traceback

import darknet
import cv2
import numpy as np
import os
import sys
import numpy as np
import torch
from sort import Sort


class Detection:

    def __init__(self, config_file, data_file, weights):
        self.cfg_filepath = config_file
        self.data_file = data_file
        self.model_weights = weights
        self.darknet_width = 416
        self.darknet_height = 416
        self.network, self.class_names, self.class_colors = darknet.load_network(config_file, data_file,
                                                                                 weights, batch_size=1)

        self.darknet_width = darknet.network_width(self.network)  # 416
        self.darknet_height = darknet.network_height(self.network)  # 416
        self.inside = 0
        self.outside = 0
        #self.sort_tracked_object=sort_tracked_object
        self.id1=[]
        self.id2=[]
        self.sort_tracked_object=[]

    def bbox2points(self,bbox):
        """
        From bounding box yolo format
        to corner points cv2 rectangle
        """
        x, y, w, h = bbox
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def convert2relative(self, bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h = bbox
        _height = self.darknet_height
        _width = self.darknet_width
        return x / _width, y / _height, w / _width, h / _height

    def convert2original(self, image, bbox):
        x, y, w, h = self.convert2relative(bbox)  # x,y,w,h are in 416

        image_h, image_w, __ = image.shape

        orig_x = int(x * image_w)
        orig_y = int(y * image_h)
        orig_width = int(w * image_w)
        orig_height = int(h * image_h)

        bbox_converted = [orig_x, orig_y, orig_width, orig_height]

        return bbox_converted

    def yoloInit(self):
        self.network, self.class_names, self.class_colors = darknet.load_network(self.cfg_filepath, self.data_file,
                                                                                 self.model_weights, batch_size=1)

        self.darknet_width = darknet.network_width(self.network)  # 416
        self.darknet_height = darknet.network_height(self.network)  # 416

    def detection(self, frame,sort_tracker):

        org_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.darknet_width, self.darknet_height), interpolation=cv2.INTER_LINEAR)

        img_for_detect = darknet.make_image(self.darknet_width, self.darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, img_for_detect, thresh=0.5)

        # (label,conf,(x1,y1,x2,y2))
        # x1 = top
        detections_adjusted = []
        detection_sort = np.empty((0, 6)) # [x1,y1,x2,y2,0.95,0]
         # 416 X416
        if frame is not None:

            try:
                for label, confidence, bbox in detections:
                    # print("bbox_adjusted : ", bbox)
                    bbox_adjusted = self.convert2original(frame, bbox)

                    detections_adjusted.append((str(label), confidence, bbox_adjusted))
                    print("@@@@@@ : ",detections_adjusted)
                for label, confidence, bbox in detections_adjusted:
                    left,top,right,bottom = self.bbox2points(bbox)
                    detection_sort = np.vstack((detection_sort,np.array([left,top,right,bottom,0.95,0])))
                print("detection_sort : ",detection_sort)
                #global sort_tracked_object
                sort_tracked_object = sort_tracker.update(detection_sort)

                print("### : ",sort_tracked_object)

                if len(sort_tracked_object)>0:
                    image = darknet.draw_custom_boxes(detections_adjusted, frame, self.class_colors,
                                                      sort_tracked_object)
                else:
                    image = darknet.draw_boxes(detections_adjusted, frame, self.class_colors)
            except Exception as e:
                traceback.print_exc()
                print(e)
                image = org_frame.copy()
        #print("image : ",image)
        return image,sort_tracked_object

    def detect(self,frame):
        #org_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.darknet_width, self.darknet_height), interpolation=cv2.INTER_LINEAR)

        img_for_detect = darknet.make_image(self.darknet_width, self.darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, img_for_detect, thresh=0.5)
        print("detections: ",detections)

        return detections

    def count_people(self, sort_object):

        if len(sort_object) > 0:

            if len(self.id1) != 0 and len(self.id2) != 0:

                if str(sort_object[0][-1])==self.id1[-1]:
                    self.id1.pop()
                    self.inside-=1

                elif str(sort_object[0][-1])==self.id2[-1]:
                    self.id2.pop()
                    self.outside-=1

            #print("line:", height//4)
            #print('id1:---- ', self.id1)

            if sort_object[0][1] > (height // 4) and str(sort_object[0][-1]) not in self.id1:

                self.inside += 1
                self.id1.append(str(sort_object[0][-1]))

            elif sort_object[0][1] < (height // 4) and str(sort_object[0][-1]) not in self.id2:
                self.outside += 1

                self.id2.append(str(sort_object[0][-1]))


        return frame





if __name__ == "__main__":

    save_video = True
    cap_path = "/Users/vivekgupta/work/PersonDetection/videos/video1.avi"
    cfg_filepath = "/Users/vivekgupta/work/darknet/persondetection_data/data/yolov3-tiny_4013.cfg"
    data_file = "/Users/vivekgupta/work/darknet/persondetection_data/data/coco.data"
    model_weights = "/Users/vivekgupta/work/darknet/persondetection_data/data/yolov3-tiny_final_4013.weights"

    cap = cv2.VideoCapture(cap_path)

    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("w : ",width)
    print("h"
          " : ",height)
    if save_video:
        out = cv2.VideoWriter('90_degree_age_demo_1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (int(width), int(height)))

    fire = Detection(cfg_filepath, data_file, model_weights)
    sort_tracker = Sort()
    fire.yoloInit()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        cv2.line(frame, (0, int(height // 4)), (int(width), int(height // 4)), (0, 255, 255), 2)
        frame_count += 1
        print(frame_count)
        #cv2.line(frame, (0,  height), (width,  height), (0, 255, 255), 2)
        if frame_count < 250:
            continue

        if not ret:
            break
        #print("$$$$$ : ",frame.shape)
        frame,sort_object = fire.detection(frame,sort_tracker)

        org=(50,50)
        fontScale=1
        font=cv2.FONT_HERSHEY_PLAIN
        color=(0,0,255)
        thickness=2
        print("$$ : ",sort_object)
        #frame = np.array(frame)
        #print(frame.shape)


        if len(sort_object) > 0:

            info = [
                ("Up", fire.inside),
                ("Down",fire.outside)

            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {} ".format(k, v)
                print(text)
                cv2.rectangle(frame, (0,0), (150,100), (0, 0, 0), -1)
                frame = cv2.putText(fire.count_people(sort_object), str(sort_object[0][-1]), org, font, fontScale,
                                    color, thickness, cv2.LINE_AA)

                i = str(fire.inside)
                frame = cv2.putText(frame, f'IN : {i}', (50, 20), font, fontScale,
                                    color, thickness, cv2.LINE_AA)
                o = str(fire.outside)
                frame = cv2.putText(frame,f'OUT : {o}', (50, 30), font, fontScale,
                                    color, thickness, cv2.LINE_AA)

        else:
            info = [
                ("Up", fire.inside),
                ("Down", fire.outside)

            ]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {} ".format(k, v)
                print(text)
                cv2.rectangle(frame, (0, 0), (150, 100), (0, 0, 0), -1)
                i = str(fire.inside)
                frame = cv2.putText(frame,f'IN : {i}' , (50, 20), font, fontScale,
                                    color, thickness, cv2.LINE_AA)
                o = str(fire.outside)
                frame = cv2.putText(frame,f'OUT : {o}', (50, 30), font, fontScale,
                                    color, thickness, cv2.LINE_AA)

        print("ID1: ", fire.id1)
        print("ID2: ", fire.id2)
        cv2.imshow('Inference', frame)



        if save_video:
            out.write(frame)

        if cv2.waitKey(1) == 27:  # Esc
            break

    cap.release()
    if save_video:
        out.release()

