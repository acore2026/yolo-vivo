"""Session presentation and startup lifecycle regression tests, no GPU required."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from aiohttp import web
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.stream_processor import FlowerProcessor
from service.stream_app import make_app


class Engine:
    def __init__(self):
        self.boxes = [dict({'class': 0}, xyxy=[20, 30, 150, 200], confidence=.876)]
    def infer(self, frame):
        return self.boxes


class ProcessorTest(unittest.TestCase):
    def test_arrival_hold_loss_rearrival_and_isolation(self):
        engine = Engine()
        now = [0.]
        a = FlowerProcessor(engine, lambda: now[0])
        b = FlowerProcessor(engine, lambda: now[0])
        frame = np.zeros((480, 640, 3), np.uint8)
        for i in range(9):
            a.process(frame, {})
        overlay = next(iter(a.overlays.tracks.values()))['overlay']
        self.assertFalse(overlay.arrived)
        with patch('service.stream_processor.cv2.putText', wraps=__import__('cv2').putText) as draw:
            output = a.process(frame, {})
            self.assertEqual(draw.call_args.args[1], 'peony 0.88')
        self.assertTrue(overlay.arrived)
        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(frame.any())
        self.assertFalse(b.overlays.tracks)
        now[0] = 100
        a.process(frame, {})
        self.assertEqual(overlay.text_at(now[0]), overlay.panel.plain_text)
        engine.boxes = []
        for _ in range(9):
            a.process(frame, {})
        self.assertTrue(overlay.arrived)
        a.process(frame, {})
        self.assertFalse(a.overlays.tracks)
        engine.boxes = Engine().boxes
        for _ in range(10):
            a.process(frame, {})
        overlay = next(iter(a.overlays.tracks.values()))['overlay']
        self.assertEqual(len(overlay.text_at(now[0])), 1)
        a.process(np.zeros((720, 1280, 3), np.uint8), {})
        self.assertFalse(next(iter(a.overlays.tracks.values()))['overlay'].arrived)
        a.close()
        with self.assertRaises(RuntimeError):
            a.process(frame, {})

    def test_reject_and_failure(self):
        engine = Engine()
        engine.boxes[0]['class'] = 3
        p = FlowerProcessor(engine)
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertFalse(p.process(frame, {}).any())
        with patch.object(engine, 'infer', side_effect=RuntimeError('model failure')):
            with self.assertRaisesRegex(RuntimeError, 'model failure'):
                p.process(frame, {})


class StartupTest(unittest.IsolatedAsyncioTestCase):
    async def test_preload_before_ready_and_failure(self):
        # Test lifecycle ordering without depending on sandbox thread wakeups.
        async def inline(func):
            return func()
        patcher = patch('service.stream_app.asyncio.to_thread', inline)
        patcher.start()
        self.addCleanup(patcher.stop)
        calls = []
        def factory(**kwargs):
            return FlowerProcessor(Engine())
        factory.prepare = lambda: calls.append('prepared')
        runner = web.AppRunner(make_app(factory))
        await runner.setup()
        self.assertEqual(calls, ['prepared'])
        await runner.cleanup()
        def broken():
            raise RuntimeError('bad weights')
        factory.prepare = broken
        runner = web.AppRunner(make_app(factory))
        with self.assertRaisesRegex(RuntimeError, 'bad weights'):
            await runner.setup()
        await runner.cleanup()


if __name__ == '__main__':
    unittest.main()
