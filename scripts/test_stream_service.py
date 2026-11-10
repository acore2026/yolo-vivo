"""Isolated transport tests; no camera, GPU, production streams or QoS writes."""
import asyncio
import json
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from aiohttp import ClientSession, WSServerHandshakeError, WSMsgType, web
from service.stream_app import make_app
from service.stream_processor import create_passthrough_processor as create_processor
from service.stream_protocol import pack_packet, unpack_packet, MAX_JPEG


def packet(fid=0, size=(480, 640), color=(30, 80, 180)):
    pixels = np.full((*size, 3), color, np.uint8)
    ok, jpeg = cv2.imencode('.jpg', pixels)
    assert ok
    return pack_packet(dict(version=1, type='frame', frame_id=fid, codec='jpeg',
                            pts=fid * 3000, time_base=[1, 90000]), jpeg.tobytes())


class StreamTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        for runner in self.runners:
            await runner.cleanup()

    async def start(self, factory=create_processor, **options):
        runner = web.AppRunner(make_app(factory, **options), shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        self.runners.append(runner)
        return 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])

    async def connect(self, base, stream='control', **options):
        ws = await self.client.ws_connect(base + '/v1/streams/' + stream + '/ws', **options)
        ready = await ws.receive_json(timeout=3)
        self.assertEqual(ready['type'], 'ready')
        return ws, ready

    async def result(self, ws):
        message = await ws.receive(timeout=3)
        self.assertEqual(message.type, WSMsgType.BINARY, message)
        header, jpeg = unpack_packet(message.data)
        pixels = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return header, pixels

    async def drained(self, base):
        for _ in range(100):
            async with self.client.get(base + '/v1/streams') as response:
                status = await response.json()
            if not status['streams']:
                return
            await asyncio.sleep(.01)
        self.fail('Session did not drain: ' + str(status))

    async def test_roundtrip_resolution_timestamp_and_two_streams(self):
        base = await self.start()
        one, ready1 = await self.connect(base, 'device-control')
        two, ready2 = await self.connect(base, 'device-dynamic-qos')
        self.assertNotEqual(ready1['session_id'], ready2['session_id'])
        for fid, shape in enumerate([(480, 640), (720, 1280), (360, 640), (480, 640)]):
            await one.send_bytes(packet(fid, shape, (15, 40, 200)))
            await two.send_bytes(packet(fid, shape, (200, 40, 15)))
            for ws, ready, color in [(one, ready1, (15, 40, 200)), (two, ready2, (200, 40, 15))]:
                head, image = await self.result(ws)
                self.assertEqual(head['frame_id'], fid)
                self.assertEqual(head['session_id'], ready['session_id'])
                self.assertEqual(head['pts'], fid * 3000)
                self.assertEqual(head['time_base'], [1, 90000])
                self.assertEqual(image.shape[:2], shape)
                self.assertLess(np.abs(image[0, 0].astype(int) - color).max(), 5)
        await one.close()
        await two.close()
        await self.drained(base)
        new, ready3 = await self.connect(base, 'device-control')
        self.assertNotEqual(ready1['session_id'], ready3['session_id'])
        await new.send_bytes(packet(0))
        head, _ = await self.result(new)
        self.assertEqual(head['frame_id'], 0)
        await new.close()

    async def test_custom_processor_and_session_state(self):
        closed = []
        class Marker:
            mode = 'test-marker'
            def __init__(self, stream_id, session_id):
                self.count = 0
                self.sid = session_id
            def process(self, frame, metadata):
                self.count += 1
                frame[:100, :100] = (0, self.count * 30, 0)
                return frame
            def close(self):
                closed.append(self.sid)
        base = await self.start(Marker)
        a, ra = await self.connect(base, 'a')
        b, rb = await self.connect(base, 'b')
        for ws, fid, value in [(a, 0, 30), (a, 1, 60), (b, 0, 30)]:
            await ws.send_bytes(packet(fid))
            head, image = await self.result(ws)
            self.assertEqual(head['processor'], 'test-marker')
            self.assertLess(abs(int(image[30, 30, 1]) - value), 4)
        await a.close()
        await b.close()
        await self.drained(base)
        self.assertEqual(set(closed), {ra['session_id'], rb['session_id']})

    async def test_auth_duplicate_stream_and_capacity(self):
        base = await self.start(token='test-secret', max_sessions=1)
        async with self.client.get(base + '/health/live') as response:
            self.assertEqual(response.status, 401)
        headers = {'Authorization': 'Bearer test-secret'}
        ws, _ = await self.connect(base, headers=headers)
        for stream, status in [('control', 409), ('another', 503)]:
            with self.assertRaises(WSServerHandshakeError) as err:
                await self.connect(base, stream, headers=headers)
            self.assertEqual(err.exception.status, status)
        await ws.close()

    async def test_bad_packets_and_corrupt_image(self):
        base = await self.start()
        head = dict(version=1, type='frame', frame_id=0, codec='jpeg')
        values = [b'bad', pack_packet(head, b'not-a-jpeg'),
                  pack_packet({**head, 'frame_id': True}, b'x'),
                  pack_packet({**head, 'pts': 123}, b'x')]
        for i, value in enumerate(values):
            ws, _ = await self.connect(base, 'invalid-' + str(i))
            await ws.send_bytes(value)
            message = await ws.receive_json(timeout=3)
            self.assertEqual(message['type'], 'error')
            self.assertEqual(message['code'], 'INVALID_FRAME')
            await ws.close()

    async def test_nonmonotonic_frame_rejected(self):
        base = await self.start()
        ws, _ = await self.connect(base)
        await ws.send_bytes(packet(2))
        await self.result(ws)
        await ws.send_bytes(packet(2))
        error = await ws.receive_json(timeout=3)
        self.assertIn('strictly increase', error['message'])
        await ws.close()

    async def test_packet_and_decoded_image_limits(self):
        from io import BytesIO
        from PIL import Image
        base = await self.start()
        ws, _ = await self.connect(base, 'too-large')
        try:
            # aiohttp may reject the advertised size before the sender finishes
            # writing the payload; either a close frame or TCP reset is valid.
            try:
                await ws.send_bytes(b'x' * (MAX_JPEG + 20000))
                message = await ws.receive(timeout=3)
                self.assertIn(message.type, (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR))
            except ConnectionError:
                pass
        finally:
            await ws.close()
        await self.drained(base)
        ws, _ = await self.connect(base, 'too-wide')
        out = BytesIO()
        Image.new('RGB', (4097, 10)).save(out, format='JPEG')
        await ws.send_bytes(pack_packet(dict(version=1, type='frame', frame_id=0, codec='jpeg'), out.getvalue()))
        self.assertEqual((await ws.receive_json(timeout=3))['code'], 'INVALID_FRAME')
        await ws.close()

    async def test_slow_processor_keeps_latest_frame_only(self):
        entered, release = threading.Event(), threading.Event()
        class Slow:
            def process(self, frame, metadata):
                if metadata['frame_id'] == 0:
                    entered.set()
                    release.wait(2)
                return frame
        base = await self.start(lambda **kw: Slow(), max_pending_age=2)
        ws, _ = await self.connect(base)
        try:
            await ws.send_bytes(packet(0))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            for fid in range(1, 11):
                await ws.send_bytes(packet(fid))
            for _ in range(100):
                async with self.client.get(base + '/v1/streams') as response:
                    status = (await response.json())['streams'][0]
                if status['received_frames'] == 11:
                    break
                await asyncio.sleep(.01)
            self.assertEqual(status['received_frames'], 11)
            self.assertEqual(status['dropped_frames'], 9)
            release.set()
            self.assertEqual((await self.result(ws))[0]['frame_id'], 0)
            self.assertEqual((await self.result(ws))[0]['frame_id'], 10)
        finally:
            release.set()
            await ws.close()

    async def test_timeout_quarantines_running_processor(self):
        release, closed = threading.Event(), threading.Event()
        class Stuck:
            def process(self, frame, metadata):
                release.wait(2)
                return frame
            def close(self):
                closed.set()
        base = await self.start(lambda **kw: Stuck(), process_timeout=.08, max_sessions=1)
        ws, _ = await self.connect(base)
        try:
            await ws.send_bytes(packet())
            self.assertEqual((await ws.receive_json(timeout=3))['code'], 'TIMEOUT')
            await ws.close()
            with self.assertRaises(WSServerHandshakeError) as err:
                await self.connect(base, 'new')
            self.assertEqual(err.exception.status, 503)
            self.assertFalse(closed.is_set())
        finally:
            release.set()
        await self.drained(base)
        self.assertTrue(closed.is_set())
        new, _ = await self.connect(base)
        await new.close()

    async def test_processing_failure_no_fake_image(self):
        class Broken:
            def process(self, frame, metadata):
                raise RuntimeError('model unavailable')
        base = await self.start(lambda **kw: Broken())
        ws, _ = await self.connect(base)
        await ws.send_bytes(packet())
        error = await ws.receive_json(timeout=3)
        self.assertEqual(error['code'], 'PROCESSING_FAILED')
        self.assertEqual(error['message'], 'model unavailable')
        await ws.close()
        await self.drained(base)

    async def test_idle_and_shutdown(self):
        base = await self.start(idle_timeout=.08)
        ws, _ = await self.connect(base)
        self.assertEqual((await ws.receive_json(timeout=3))['code'], 'IDLE_TIMEOUT')
        await ws.close()
        await self.drained(base)
        base2 = await self.start()
        active, _ = await self.connect(base2)
        await self.runners[-1].cleanup()
        self.runners.pop()
        self.assertIn((await active.receive(timeout=3)).type, (WSMsgType.CLOSE, WSMsgType.CLOSED))
        await active.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
