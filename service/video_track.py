"""WebRTC track for the latest annotated car-detection frame."""
import asyncio
from fractions import Fraction
import time

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame


class OutputTrack(VideoStreamTrack):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.origin = time.monotonic()
        self.last = 0

    async def recv(self):
        await asyncio.sleep(max(0, self.last + 1 / 30 - time.monotonic()))
        self.last = time.monotonic()
        item = self.pipeline.rendered
        if (item and self.last - item[1] < .5
                and self.pipeline.status not in ('STOPPED', 'STOPPING', 'ERROR')):
            frame = item[2]
        else:
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, 'NO FRESH VIDEO', (160, 240), 0, 1,
                        (255, 255, 255), 2)
        out = VideoFrame.from_ndarray(frame, format='bgr24')
        out.pts = int((self.last - self.origin) * 90000)
        out.time_base = Fraction(1, 90000)
        return out
