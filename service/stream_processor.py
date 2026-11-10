"""Per-session presentation with a shared, serialized three-flower model.

The transport calls process sequentially per session and close after it drains.
No camera is opened here. Inference failures propagate to the transport.
"""
import time
import cv2
from .flower_runtime import runtime
from .three_flower_overlay import ThreeFlowerOverlays

NAMES = ('peony', 'lily', 'chrysanthemum')


class PassthroughProcessor:
    mode = 'passthrough'

    def process(self, frame, metadata):
        return frame

    def close(self):
        pass


class FlowerProcessor:
    mode = 'three_flowers_introductions'

    def __init__(self, engine=runtime, clock=time.monotonic):
        self.engine = engine
        self.clock = clock
        self.overlays = ThreeFlowerOverlays()
        self.shape = None
        self.closed = False

    def process(self, frame, metadata):
        if self.closed:
            raise RuntimeError('Processor is closed')
        detections = self.engine.infer(frame)
        if self.shape != frame.shape:
            self.overlays = ThreeFlowerOverlays()
            self.shape = frame.shape
        now = self.clock()
        kept = self.overlays.update(detections, now)
        output = frame.copy()
        for d in kept:
            x1, y1, x2, y2 = map(round, d['xyxy'])
            cv2.rectangle(output, (x1, y1), (x2, y2), (90, 225, 240), 2)
            cv2.putText(output, f"{NAMES[d['class']]} {d['confidence']:.2f}",
                        (max(0, x1), max(22, y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, (90, 225, 240), 2)
        return self.overlays.draw(output, now)

    def close(self):
        self.closed = True
        self.overlays = ThreeFlowerOverlays()


def create_processor(*, stream_id, session_id):
    return FlowerProcessor()


# Optional lifecycle hooks understood by stream_app; no loading at import time.
create_processor.prepare = runtime.prepare
create_processor.status = runtime.status


def create_passthrough_processor(*, stream_id, session_id):
    return PassthroughProcessor()
