"""Per-object flower introductions with ten-frame debounce and video-time typing."""
from pathlib import Path
import time
from PIL import ImageFont
from .car_overlay import ArrivalOverlay
from .markdown_panel import MarkdownPanel

INTRODUCTIONS = {
    0: '## 芍药 · 花开有约\n层叠的花瓣搭配**柔和的粉色**，营造温柔、饱满的观赏效果。\n\n> 花语：美丽与惜别。',
    1: '## 兰花 · 雅致相伴\n舒展的花姿与**清雅的气质**，为桌面增添一份灵动。\n\n> 花语：高洁与美好。',
    2: '## 向日葵 · 向光而生\n明亮的黄色让空间充满活力，传递**积极、温暖**的心情。\n\n> 花语：阳光与希望。',
    3: '## 桉树叶 · 一抹清新\n灰绿色的圆润叶片，带来**轻盈、自然**的层次感，适合点缀简洁的空间。\n\n> 一抹绿意，点亮日常。',
}

class FlowerIntroduction(ArrivalOverlay):
    CHARACTER_SECONDS = .04

    def __init__(self, class_id):
        self.reset()
        self.panel = MarkdownPanel(INTRODUCTIONS[class_id])
        fonts = Path(__file__).with_name('fonts')
        self.panel.font = ImageFont.truetype(str(fonts/'NotoSansCJKsc-Flowers.otf'), 14)
        self.panel.bold_font = ImageFont.truetype(str(fonts/'NotoSansCJKsc-Flowers-Bold.otf'), 14)
        self.panel.heading_font = ImageFont.truetype(str(fonts/'NotoSansCJKsc-Flowers-Bold.otf'), 18)
        self.font = self.panel.font

    def text_at(self, now=None):
        """Type once per arrival, then retain the complete introduction."""
        if not self.arrived:
            return None
        now = time.monotonic() if now is None else now
        elapsed = max(0., now - self.started_at)
        message = self.panel.plain_text
        count = min(len(message), int(elapsed / self.CHARACTER_SECONDS) + 1)
        return message[:count]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


class FlowerOverlays:
    """Greedy same-class IoU matching; lightweight tracking, not re-identification."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now):
        # Suppress nested duplicate boxes for presentation only. Detector logs
        # and inference decisions remain available separately.
        kept = []
        for d in sorted(detections, key=lambda d: -d['confidence']):
            if d['class'] in INTRODUCTIONS and all(d['class'] != k['class'] or iou(d['xyxy'], k['xyxy']) < .5 for k in kept):
                kept.append(d)
        pairs = sorted([(iou(t['box'], d['xyxy']), ident, j)
                        for ident, t in self.tracks.items() for j, d in enumerate(kept)
                        if t['class'] == d['class']], reverse=True)
        matched_tracks, matched_detections = set(), set()
        for score, ident, j in pairs:
            if score < .15 or ident in matched_tracks or j in matched_detections:
                continue
            t = self.tracks[ident]; d = kept[j]
            t['box'] = d['xyxy']; t['overlay'].update([{'bbox_xyxy': d['xyxy']}], now=now)
            matched_tracks.add(ident); matched_detections.add(j)
        for ident in list(self.tracks):
            if ident not in matched_tracks:
                overlay = self.tracks[ident]['overlay']; overlay.update([], now=now)
                if overlay.missing_frames >= overlay.REQUIRED_FRAMES:
                    del self.tracks[ident]
        for j, d in enumerate(kept):
            if j not in matched_detections:
                overlay = FlowerIntroduction(d['class']);overlay.update([{'bbox_xyxy': d['xyxy']}], now=now)
                self.tracks[self.next_id] = {'class': d['class'], 'box': d['xyxy'], 'overlay': overlay}
                self.next_id += 1
        return kept

    def draw(self, frame, now):
        for track in self.tracks.values():
            track['overlay'].draw(frame, now=now)
        return frame
