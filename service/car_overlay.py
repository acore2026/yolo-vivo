"""Debounced, growing text to the right of each car detection box."""
import time

from .markdown_panel import MarkdownPanel


class ArrivalOverlay:
    REQUIRED_FRAMES = 10
    MESSAGE = (
        '救护车已到达！\n\n'
        '## 现场状态\n'
        '**机器车已就位**，请保持通道畅通。\n\n'
        '- 检测状态：**稳定识别**\n'
        '- 视频画面：持续更新\n'
        '- 到达确认：`连续 10 帧`\n\n'
        '> 请等待工作人员接应，避免靠近行驶中的机器车。\n\n'
        '---\n'
        '### 接应流程\n'
        '1. 确认车辆停稳\n'
        '2. 核对本次配送任务\n'
        '3. 按照现场指引完成交接\n\n'
        '*温馨提示*：系统会持续观察车辆位置。短暂漏检时保留提示，'
        '连续十帧未检测到车辆后自动隐藏，再次确认到达时从头开始显示。\n\n'
        '[查看接应说明](https://example.com/arrival)'
    )
    CHARACTER_SECONDS = .1
    HOLD_SECONDS = 2.
    BOX_GAP = 8
    BOX_WIDTH = MarkdownPanel.WIDTH

    def __init__(self):
        self.reset()
        self.panel = MarkdownPanel(self.MESSAGE)
        self.font = self.panel.font

    def reset(self):
        self.arrived = False
        self.detected_frames = 0
        self.missing_frames = 0
        self.started_at = None
        self.boxes = []
        self._cached_text = None
        self._cached_banner = None

    def update(self, detections, now=None):
        if detections:
            self.boxes = [tuple(item['bbox_xyxy']) for item in detections]
            self.detected_frames = min(self.detected_frames + 1, self.REQUIRED_FRAMES)
            self.missing_frames = 0
            if self.detected_frames == self.REQUIRED_FRAMES and not self.arrived:
                self.arrived = True
                self.started_at = time.monotonic() if now is None else now
        else:
            self.missing_frames = min(self.missing_frames + 1, self.REQUIRED_FRAMES)
            self.detected_frames = 0
            if self.missing_frames == self.REQUIRED_FRAMES:
                self.arrived = False
                self.started_at = None
                self.boxes = []

    def text_at(self, now=None):
        if not self.arrived:
            return None
        now = time.monotonic() if now is None else now
        elapsed = max(0., now - self.started_at)
        message = self.panel.plain_text
        cycle = len(message) * self.CHARACTER_SECONDS + self.HOLD_SECONDS
        count = min(len(message), int((elapsed % cycle) / self.CHARACTER_SECONDS) + 1)
        return message[:count]

    def _banner_for(self, text):
        if text != self._cached_text:
            self._cached_banner = self.panel.render(len(text))
            self._cached_text = text
        return self._cached_banner

    def draw(self, frame, now=None):
        text = self.text_at(now)
        if text is None:
            return frame
        tile = self._banner_for(text)
        height, width = tile.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        for _, y1, x2, _ in self.boxes:
            x, y = round(x2) + self.BOX_GAP, round(y1)
            # Keep the anchor fixed relative to the box; clip rather than shift.
            left, top = max(0, x), max(0, y)
            right, bottom = min(frame_width, x + width), min(frame_height, y + height)
            if right > left and bottom > top:
                frame[top:bottom, left:right] = tile[top - y:bottom - y, left - x:right - x]
        return frame
