"""Reference client: real JPEG round trips, resolution changes and two streams."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from aiohttp import ClientSession, WSMsgType
from service.stream_protocol import MAX_PACKET, pack_packet, unpack_packet


async def probe(base, count):
    token = os.environ.get('STREAM_API_TOKEN')
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    async with ClientSession(headers=headers) as client:
        async def stream(name, color):
            rtts, server_times, first = [], [], time.monotonic()
            async with client.ws_connect(base.rstrip('/') + '/v1/streams/' + name + '/ws', max_msg_size=MAX_PACKET) as ws:
                ready = await ws.receive_json(timeout=5)
                assert ready['type'] == 'ready', ready
                for fid in range(count):
                    h, w = (480, 640) if fid < count // 2 else (720, 1280)
                    pixels = np.full((h, w, 3), color, np.uint8)
                    cv2.putText(pixels, name + ' ' + str(fid), (20, 80), 0, 1, (255, 255, 255), 2)
                    ok, jpeg = cv2.imencode('.jpg', pixels)
                    assert ok
                    start = time.monotonic()
                    await ws.send_bytes(pack_packet(dict(version=1, type='frame', codec='jpeg',
                        frame_id=fid, pts=fid * 3000, time_base=[1, 90000]), jpeg.tobytes()))
                    reply = await ws.receive(timeout=5)
                    assert reply.type == WSMsgType.BINARY, reply
                    header, output = unpack_packet(reply.data)
                    rtts.append((time.monotonic() - start) * 1000)
                    decoded = cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR)
                    assert header['session_id'] == ready['session_id'] and header['stream_id'] == name
                    assert header['frame_id'] == fid and header['pts'] == fid * 3000
                    assert header['time_base'] == [1, 90000]
                    assert decoded is not None and decoded.shape == pixels.shape
                    assert header['width'] == w and header['height'] == h
                    # The default processor is passthrough; verify actual pixels, not just counters.
                    if header['processor'] == 'passthrough':
                        assert np.abs(decoded[-20, -20].astype(int) - color).max() < 5
                    server_times.append(header['server_total_ms'])
                return dict(stream=name, returned_frames=count, resolutions=['640x480', '1280x720'],
                            roundtrip_ms_median=round(statistics.median(rtts), 2),
                            roundtrip_ms_max=round(max(rtts), 2),
                            server_ms_median=round(statistics.median(server_times), 2),
                            elapsed_s=round(time.monotonic() - first, 2))
        results = await asyncio.gather(stream('probe-control', (20, 60, 200)),
                                       stream('probe-dynamic-qos', (200, 60, 20)))
        print(json.dumps({'passed': True, 'streams': results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8767')
    parser.add_argument('--frames', type=int, default=30)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error('--frames must be at least 2')
    asyncio.run(probe(args.url, args.frames))
