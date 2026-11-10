"""Latest-frame robot car detection with a debounced arrival banner."""
import hashlib
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .car_overlay import ArrivalOverlay


class CarPipeline:
    def __init__(self, model, source, device='cpu', confidence=.25, camera_id='hospital-camera'):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.model = YOLO(model)
        if self.model.task != 'detect' or self.model.names != {0: 'ambulance_robot_car'}:
            raise ValueError('Expected single-class ambulance_robot_car detection model')
        self.version = hashlib.sha256(Path(model).read_bytes()).hexdigest()
        self.source, self.device, self.confidence, self.camera_id = source, device, confidence, camera_id
        self.condition = threading.Condition()
        self.halt = threading.Event()
        self.threads = []
        self.status, self.error = 'STOPPED', None
        self.latest = self.rendered = self.result = None
        self.sequence = self.processed = self.dropped = 0
        self.last_result = 0
        self.done = False
        self.arrival_overlay = ArrivalOverlay()

    def start(self):
        if any(t.is_alive() for t in self.threads):
            return
        self.halt.clear()
        self.done = False
        self.latest = self.rendered = self.result = None
        self.sequence = self.processed = self.dropped = 0
        self.last_result = 0
        self.arrival_overlay.reset()
        self.error = None
        self.status = 'STARTING'
        try:
            self.model.predict(np.zeros((480,640,3),np.uint8),device=self.device,verbose=False)
        except Exception as exc:
            self.status, self.error = 'ERROR', str(exc)
            raise
        self.threads = [threading.Thread(target=f,daemon=True) for f in (self._capture,self._infer)]
        for t in self.threads:
            t.start()

    def stop(self):
        self.halt.set()
        with self.condition:
            self.condition.notify_all()
        for t in self.threads:
            t.join(timeout=3)
        self.status = 'STOPPING' if any(t.is_alive() for t in self.threads) else 'STOPPED'

    def _capture(self):
        usb = self.source.startswith('/dev/video')
        network = '://' in self.source
        try:
            while not self.halt.is_set():
                cap = cv2.VideoCapture(self.source,cv2.CAP_V4L2 if usb else cv2.CAP_ANY)
                try:
                    if not cap.isOpened():
                        raise RuntimeError('Cannot open configured camera/video source')
                    if usb:
                        for prop,value in ((3,640),(4,480),(5,30),(cv2.CAP_PROP_BUFFERSIZE,1)):
                            cap.set(prop,value)
                    fps = cap.get(5)
                    fps = fps if 0 < fps <= 120 else 30
                    origin, count = time.monotonic(), 0
                    while not self.halt.is_set():
                        if not usb and not network:
                            if self.halt.wait(max(0,origin+count/fps-time.monotonic())):
                                break
                        ok, frame = cap.read()
                        if not ok:
                            self.status = 'UNAVAILABLE' if usb or network else 'EOF'
                            break
                        captured, wall = time.monotonic(), time.time()
                        count += 1
                        h,w = frame.shape[:2]
                        ratio = min(640/w,480/h)
                        nw,nh = max(1,round(w*ratio)),max(1,round(h*ratio))
                        x,y = (640-nw)//2,(480-nh)//2
                        canvas = np.full((480,640,3),114,np.uint8)
                        canvas[y:y+nh,x:x+nw] = cv2.resize(frame,(nw,nh))
                        with self.condition:
                            self.sequence += 1
                            self.latest = (self.sequence,captured,wall,canvas)
                            self.condition.notify_all()
                    if not usb and not network:
                        break
                finally:
                    cap.release()
                self.halt.wait(1)
        except Exception as exc:
            self.status,self.error = 'ERROR',str(exc)
            self.halt.set()
        finally:
            self.done = True
            with self.condition:
                self.condition.notify_all()

    def _infer(self):
        previous = 0
        try:
            while not self.halt.is_set():
                with self.condition:
                    self.condition.wait_for(lambda:self.halt.is_set() or self.done or (self.latest and self.latest[0]>previous),timeout=.25)
                    if self.halt.is_set():
                        break
                    if not self.latest or self.latest[0] <= previous:
                        if self.done:
                            break
                        continue
                    seq,captured,wall,frame = self.latest
                self.dropped += max(0,seq-previous-1)
                previous = seq
                if time.monotonic()-captured > .5:
                    continue
                pred = self.model.predict(frame,device=self.device,imgsz=640,conf=self.confidence,iou=.5,verbose=False)[0]
                if self.halt.is_set() or time.monotonic()-captured > .5:
                    continue
                boxes = [dict(class_id=int(b.cls[0]),label='ambulance_robot_car',confidence=float(b.conf[0]),bbox_xyxy=b.xyxy[0].tolist()) for b in pred.boxes]
                self.result = dict(type='CAR_DETECTION_RESULT',camera_id=self.camera_id,frame_id=seq,captured_at=wall,processed_at=time.time(),width=640,height=480,detections=boxes,detected=bool(boxes),simulated=True,detector_version=self.version)
                self.arrival_overlay.update(boxes)
                self.rendered = (seq,captured,self.arrival_overlay.draw(pred.plot()))
                self.last_result = captured
                self.processed += 1
                if not self.done and self.status not in ('EOF','ERROR','UNAVAILABLE'):
                    self.status = 'READY'
        except Exception as exc:
            self.status,self.error = 'ERROR',str(exc)
            self.halt.set()

    def detection(self):
        fresh = self.status == 'READY' and time.monotonic()-self.last_result < .5
        return dict(self.result or {},fresh=fresh,detected=self.result['detected'] if fresh else None,detections=self.result['detections'] if fresh else [])

    def snapshot(self):
        return dict(status=self.status,error=self.error,camera_id=self.camera_id,captured=self.sequence,processed=self.processed,dropped=self.dropped,fresh=self.detection()['fresh'],result_age_ms=(time.monotonic()-self.last_result)*1000 if self.last_result else None)
