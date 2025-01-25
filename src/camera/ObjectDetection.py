import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator, LabelAnnotator, TraceAnnotator, ByteTrack
from pytesseract import pytesseract
from PIL import Image

from src.TextDetect import DetectText
from src.tts import Voice
from src.GPT import GPTClient

import streamlit as st
import tempfile

class ObjectDetection:
    def __init__(self, captureIndex, width, height, track):
        self.width, self.height = width, height
        self.captureIndex = captureIndex
        self.boxAnnotator = BoxAnnotator()
        self.labelAnnotator = LabelAnnotator()
        self.traceAnnotator = TraceAnnotator()
        self.tracker = ByteTrack()
        self.detections = None
        self.track = track

        self.area = self.width * self.height

        self.model = YOLO("yolov8n.pt")  # Loads a pretrained model        

        # Voice Detect:
        self.speech = Voice(language='en', speed=False)

        # GPT Client:
        self.gpt_client = GPTClient()

    def callback(self, frame):
        res = self.model(frame)[0]
        detections = Detections.from_ultralytics(res)
        detections = self.tracker.update_with_detections(detections)
        detections = detections[(detections.area / self.area) < 0.3]

        # Object labeling
        labels = [
            f"#{trackerID} {res.names[classID]}"
            for classID, trackerID in zip(detections.class_id, detections.tracker_id)
        ]

        # Annotates the image
        annotatedImage = self.boxAnnotator.annotate(
            scene=frame.copy(), detections=detections
        )
        
        # Labels the image
        annotatedImage = self.labelAnnotator.annotate(
            scene=annotatedImage, detections=detections, labels=labels
        )

        # Returns the image with path traveled traced out
        if self.track: 
            return self.traceAnnotator.annotate(annotatedImage, detections=detections)
        return annotatedImage
    
    def click(self, event, param): 
        # to check if left mouse button was clicked 
        if event: 
            print("left click") 
            imgName = f"captures/{self.imgIDX}.png"


    def __call__(self):
        video = cv2.VideoCapture(self.captureIndex)
        assert video.isOpened()

        video.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Web application
        #st.title("TLDR: A Guide to Intstructions")

        frame_placeholder = st.empty()

        capture_button, ask_button, stop_button_pressed = st.columns([1,1,1])

        with capture_button: 
            b1=st.button("Capture")
        with ask_button: 
            b2=st.button("Ask")
        with stop_button_pressed: 
            b3=st.button("Stop")

        while True: 
            ti = time.time()  # Starting time

            ret, frame = video.read()  # Captures frame by frame 
            if not ret: 
                break  # breaks if read incorrectly

            image, tf = self.callback(frame), time.time()
            fps = 1 / np.round(tf - ti, 2)

            # FPS text on screen
            cv2.putText(
                img=image, text=f'FPS: {int(fps)}', org=(20, 70), fontFace=5, fontScale=1.5, color=(227, 111, 179), thickness=2
            )

            print((capture_button, ask_button))
            if b1:
                print("ok")

                self.click(True, image)
            elif b2:
                self.click(True, image)


            if cv2.waitKey(1) & 0xFF == ord(" ") or b3: 
                break

        # Release the capture
        video.release()
        cv2.destroyAllWindows()

# Create an instance of ObjectDetection
#detector = ObjectDetection(0, 1280, 720, True)
#detector()
