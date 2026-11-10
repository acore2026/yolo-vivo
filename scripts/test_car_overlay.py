"""Offline arrival-banner tests: no model, camera or HTTP listener required."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.car_overlay import ArrivalOverlay
from service.car_pipeline import CarPipeline
from service.markdown_panel import MarkdownPanel, parse_markdown


class ArrivalOverlayTests(unittest.TestCase):
    def setUp(self):
        self.overlay = ArrivalOverlay()

    def feed(self, detected, count):
        for _ in range(count):
            self.overlay.update([{'bbox_xyxy': [100, 90, 200, 300]}] if detected else [])

    def test_show_on_tenth_detection_and_hide_on_tenth_miss(self):
        self.feed(True, 9)
        self.assertFalse(self.overlay.arrived)
        self.feed(True, 1)
        self.assertTrue(self.overlay.arrived)
        self.feed(False, 9)
        self.assertTrue(self.overlay.arrived)
        self.feed(False, 1)
        self.assertFalse(self.overlay.arrived)
        self.feed(True, 10)
        self.assertTrue(self.overlay.arrived)

    def test_miss_breaks_detection_streak(self):
        self.feed(True, 9)
        self.feed(False, 1)
        self.feed(True, 9)
        self.assertFalse(self.overlay.arrived)
        self.feed(True, 1)
        self.assertTrue(self.overlay.arrived)

    def test_detection_breaks_missing_streak(self):
        self.feed(True, 10)
        self.feed(False, 9)
        self.feed(True, 1)
        self.feed(False, 9)
        self.assertTrue(self.overlay.arrived)
        self.feed(False, 1)
        self.assertFalse(self.overlay.arrived)

    def test_alternating_detections_never_arrive(self):
        for _ in range(20):
            self.feed(True, 1)
            self.feed(False, 1)
        self.assertFalse(self.overlay.arrived)

    def test_text_accumulates_and_box_grows_from_fixed_anchor(self):
        with patch('service.car_overlay.time.monotonic', return_value=10.):
            self.feed(True, 10)
        previous_height = 0
        for index in range(10):
            now = 10. + index * self.overlay.CHARACTER_SECONDS + .01
            text = self.overlay.panel.plain_text[:index + 1]
            self.assertEqual(self.overlay.text_at(now), text)
            frame = self.overlay.draw(np.zeros((480, 640, 3), np.uint8), now)
            height, width = self.overlay._cached_banner.shape[:2]
            self.assertEqual(width, 220)
            self.assertGreaterEqual(height, previous_height)
            previous_height = height
            np.testing.assert_array_equal(frame[90:90 + height, 208:208 + width], self.overlay._cached_banner)
            self.assertEqual(np.count_nonzero(np.any(frame, axis=2)), width * height)
        self.assertGreater(previous_height, 52)

    def test_markdown_is_parsed_before_typing(self):
        blocks = parse_markdown('## 标题\n**加粗**和`代码`\n- 列表\n> 引用\n1. 步骤\n---\n```text\n原样内容\n```')
        self.assertEqual([b.kind for b in blocks],
                         ['heading', 'paragraph', 'list', 'quote', 'list', 'rule', 'code'])
        self.assertIn(('加粗', 'bold'), blocks[1].runs)
        self.assertIn(('代码', 'code'), blocks[1].runs)
        self.assertNotIn('**', self.overlay.panel.plain_text)
        self.assertNotIn('https://', self.overlay.panel.plain_text)

    def test_markdown_panel_wraps_and_gradient_is_not_flat(self):
        panel = MarkdownPanel('## 标题\n' + '救护车已到达！' * 40)
        first = panel.render(1)
        full = panel.render(len(panel.plain_text))
        self.assertEqual(first.shape[1], 220)
        self.assertEqual(full.shape[1], 220)
        self.assertGreater(full.shape[0], first.shape[0])
        self.assertGreater(full.shape[0], 480)
        self.assertFalse(np.array_equal(full[5, 20], full[5, 200]))

    def test_completed_text_holds_then_loops(self):
        with patch('service.car_overlay.time.monotonic', return_value=10.):
            self.feed(True, 10)
        duration = len(self.overlay.panel.plain_text) * self.overlay.CHARACTER_SECONDS
        self.assertEqual(self.overlay.text_at(10. + duration + 1.), self.overlay.panel.plain_text)
        self.assertEqual(self.overlay.text_at(10. + duration + self.overlay.HOLD_SECONDS + .01), '救')

    def test_brief_misses_keep_progress_but_ten_misses_restart(self):
        with patch('service.car_overlay.time.monotonic', return_value=10.):
            self.feed(True, 10)
        self.feed(False, 9)
        self.assertEqual(self.overlay.text_at(10.21), '救护车')
        self.assertEqual(self.overlay.boxes, [(100, 90, 200, 300)])
        self.feed(True, 1)
        self.assertEqual(self.overlay.started_at, 10.)
        self.feed(False, 10)
        self.assertIsNone(self.overlay.text_at(11.))
        self.assertIsNone(self.overlay.started_at)
        self.assertEqual(self.overlay.boxes, [])
        with patch('service.car_overlay.time.monotonic', return_value=20.):
            self.feed(True, 9)
            self.assertIsNone(self.overlay.text_at(20.))
            self.feed(True, 1)
        self.assertEqual(self.overlay.text_at(20.), '救')
        self.assertEqual(self.overlay.text_at(20.11), '救护')

    def test_text_follows_right_edge_and_clips_without_repositioning(self):
        self.feed(True, 10)
        detections = [{'bbox_xyxy': [300, 300, 400, 400]}]
        self.overlay.update(detections)
        frame = self.overlay.draw(np.zeros((480, 640, 3), np.uint8), self.overlay.started_at)
        np.testing.assert_array_equal(frame[300:352, 408:628], self.overlay._cached_banner)
        self.assertFalse(frame[:100].any())
        self.overlay.update([{'bbox_xyxy': [500, 470, 610, 480]}])
        frame = self.overlay.draw(np.zeros((480, 640, 3), np.uint8), self.overlay.started_at)
        np.testing.assert_array_equal(frame[470:480, 618:640], self.overlay._cached_banner[:10, :22])
        self.assertEqual(np.count_nonzero(np.any(frame, axis=2)), 220)
        self.overlay.update([{'bbox_xyxy': [500, 300, 640, 400]}])
        frame = self.overlay.draw(np.zeros((480, 640, 3), np.uint8), self.overlay.started_at)
        self.assertFalse(frame.any())

    def test_long_text_is_cropped_at_screen_edge(self):
        self.feed(True, 10)
        now = self.overlay.started_at + len(self.overlay.panel.plain_text) * self.overlay.CHARACTER_SECONDS - .01
        frame = self.overlay.draw(np.zeros((480, 640, 3), np.uint8), now)
        self.assertEqual(self.overlay._cached_banner.shape[1], 220)
        self.assertGreater(self.overlay._cached_banner.shape[0], 480)
        np.testing.assert_array_equal(frame[90:, 208:428], self.overlay._cached_banner[:390])
        self.assertFalse(frame[:, :208].any())
        self.assertFalse(frame[:, 428:].any())

    def test_rendering_does_not_advance_streak(self):
        self.feed(True, 9)
        frame = np.zeros((480, 640, 3), np.uint8)
        for _ in range(30):
            np.testing.assert_array_equal(self.overlay.draw(frame.copy()), frame)
        self.assertEqual(self.overlay.detected_frames, 9)
        self.feed(True, 1)
        rendered = self.overlay.draw(frame.copy())
        self.assertEqual(rendered.shape, frame.shape)
        self.assertEqual(rendered.dtype, frame.dtype)
        self.assertTrue(np.any(rendered != frame))
        np.testing.assert_array_equal(rendered[154:], frame[154:])
        self.feed(False, 10)
        np.testing.assert_array_equal(self.overlay.draw(frame.copy()), frame)

    def test_start_resets_arrival_and_streaks(self):
        self.feed(True, 10)
        pipe = CarPipeline.__new__(CarPipeline)
        pipe.arrival_overlay = self.overlay
        pipe.threads = []
        pipe.halt = threading.Event()
        pipe.model = Mock()
        pipe.device = 'cpu'
        with patch('service.car_pipeline.threading.Thread') as thread:
            pipe.start()
            self.assertEqual(thread.call_count, 2)
        self.assertFalse(self.overlay.arrived)
        self.assertEqual(self.overlay.detected_frames, 0)
        self.assertEqual(self.overlay.missing_frames, 0)

    def test_inference_publishes_banner_at_exact_boundaries(self):
        pipe = CarPipeline.__new__(CarPipeline)
        pipe.arrival_overlay = self.overlay
        pipe.condition = threading.Condition()
        pipe.halt = threading.Event()
        pipe.done = False
        pipe.dropped = pipe.processed = 0
        pipe.status = 'STARTING'
        pipe.device, pipe.confidence = 'cpu', .25
        pipe.camera_id, pipe.version = 'test', 'test'
        pipe.rendered = None
        frame = np.zeros((480, 640, 3), np.uint8)
        pipe.latest = (1, 1., 1., frame)
        published = []
        box = SimpleNamespace(cls=[0], conf=[.9], xyxy=np.array([[1, 2, 30, 40]]))

        def predict(*args, **kwargs):
            if pipe.rendered is not None:
                published.append(bool(np.any(pipe.rendered[2])))
            seq = pipe.latest[0]
            result = Mock(boxes=[box] if seq <= 10 else [])
            result.plot.return_value = frame.copy()
            if seq < 20:
                pipe.latest = (seq + 1, 1., 1., frame)
            else:
                pipe.done = True
            return [result]

        pipe.model = Mock(predict=predict)
        with patch('service.car_pipeline.time.monotonic', return_value=1.):
            pipe._infer()
        published.append(bool(np.any(pipe.rendered[2])))
        self.assertEqual(pipe.processed, 20)
        self.assertEqual(published, [False] * 9 + [True] * 10 + [False])
        self.assertNotEqual(pipe.status, 'ERROR')


if __name__ == '__main__':
    unittest.main()
