"""Exercise HTTP frame handlers without binding a network port."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.car_app import make_app


class HTTPPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pipe = SimpleNamespace(status='READY', rendered=(42, time.monotonic(),
            np.full((480, 640, 3), (20, 60, 180), np.uint8)),
            snapshot=lambda: {'status': self.pipe.status})
        self.app = make_app(self.pipe, token='test')
        # The restricted test runtime cannot wake an event loop via sockets.
        async def inline(func, *args):
            return func(*args)
        worker = patch('service.car_app.asyncio.to_thread', side_effect=inline)
        worker.start()
        self.addCleanup(worker.stop)

    async def request(self, token='test'):
        request = make_mocked_request('GET', '/v1/video/frame.jpg',
            headers={'Authorization': 'Bearer ' + token}, app=self.app)
        route = await self.app.router.resolve(request)
        return await self.app.middlewares[0](request, route.handler)

    async def test_frame_is_decodable_and_not_cached(self):
        response = await self.request()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, 'image/jpeg')
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(response.headers['X-Frame-Id'], '42')
        frame = cv2.imdecode(np.frombuffer(response.body, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertLess(np.max(np.abs(frame.astype(int) - self.pipe.rendered[2])), 5)

    async def test_missing_stale_and_stopped_frames_are_rejected(self):
        original = self.pipe.rendered
        for item, status in [(None, 'STARTING'),
                             ((42, time.monotonic() - 1, original[2]), 'READY'),
                             (original, 'STOPPED'), (original, 'ERROR')]:
            self.pipe.rendered, self.pipe.status = item, status
            self.assertEqual((await self.request()).status, 503)

    async def test_authentication_applies_to_frames(self):
        with self.assertRaises(web.HTTPUnauthorized):
            await self.request('wrong')


if __name__ == '__main__':
    unittest.main()
