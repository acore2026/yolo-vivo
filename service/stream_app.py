"""Session-isolated JPEG WebSocket relay with a replaceable frame processor."""
import argparse
import asyncio
import hmac
import importlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np
from aiohttp import web, WSMsgType

from .stream_protocol import (MAX_PACKET, ProtocolError, decode_jpeg,
                              pack_packet, unpack_packet, validate_frame)

LOG = logging.getLogger(__name__)


@dataclass
class Session:
    stream_id: str
    ws: web.WebSocketResponse
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pending: object = None
    available: asyncio.Event = field(default_factory=asyncio.Event)
    processor: object = None
    inflight: object = None
    worker: object = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_id: int = -1
    received: int = 0
    sent: int = 0
    dropped: int = 0
    last_output: float = 0
    error: object = None
    closing: bool = False

    def snapshot(self):
        age = time.monotonic() - self.last_output if self.last_output else None
        return dict(stream_id=self.stream_id, session_id=self.session_id,
                    state='draining' if self.closing else 'connected',
                    received_frames=self.received, output_frames=self.sent,
                    dropped_frames=self.dropped, last_frame_id=self.last_id,
                    fresh=not self.closing and age is not None and age < 1,
                    output_age_ms=round(age * 1000, 1) if age is not None else None,
                    last_error=self.error)


def make_app(factory, *, token='', max_sessions=4, process_timeout=2.,
             idle_timeout=15., max_pending_age=.5, send_timeout=2., jpeg_quality=85,
             processor_name='custom'):
    sessions, disposal = {}, set()
    stopping = False
    prepared = False

    @web.middleware
    async def auth(request, handler):
        if token and not hmac.compare_digest(request.headers.get('Authorization', '').encode(),
                                             ('Bearer ' + token).encode()):
            return web.json_response({'error': 'UNAUTHORIZED'}, status=401)
        try:
            return await handler(request)
        except web.HTTPException as exc:
            return web.json_response({'error': exc.reason}, status=exc.status)

    app = web.Application(middlewares=[auth], client_max_size=MAX_PACKET)

    async def send(s, data, binary=False):
        async def write():
            async with s.send_lock:
                if binary:
                    await s.ws.send_bytes(data)
                else:
                    await s.ws.send_json(data)
        await asyncio.wait_for(write(), send_timeout)

    async def fail(s, code, message, frame_id=None):
        s.error = dict(type='error', code=code, message=message,
                       stream_id=s.stream_id, session_id=s.session_id, frame_id=frame_id)
        try:
            await send(s, s.error)
        except (ConnectionError, asyncio.TimeoutError, RuntimeError):
            pass
        await s.ws.close(code=1011 if code == 'PROCESSING_FAILED' else 1008)

    def process(s, header, jpeg, received_at):
        begin = time.monotonic()
        frame = decode_jpeg(jpeg)
        if s.processor is None:
            s.processor = factory(stream_id=s.stream_id, session_id=s.session_id)
        metadata = {**header, 'stream_id': s.stream_id, 'session_id': s.session_id}
        start = time.monotonic()
        output = s.processor.process(frame, metadata)
        processing_ms = (time.monotonic() - start) * 1000
        if not isinstance(output, np.ndarray) or output.dtype != np.uint8 or output.shape != frame.shape:
            raise ValueError('Processor must return BGR uint8 with the input dimensions')
        ok, encoded = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise ValueError('JPEG output encoding failed')
        result = dict(version=1, type='result', stream_id=s.stream_id,
                      session_id=s.session_id, frame_id=header['frame_id'], codec='jpeg',
                      width=output.shape[1], height=output.shape[0],
                      processor=getattr(s.processor, 'mode', processor_name),
                      processing_ms=round(processing_ms, 3),
                      server_total_ms=round((time.monotonic() - received_at) * 1000, 3),
                      queue_ms=round((begin - received_at) * 1000, 3))
        for key in ('pts', 'time_base'):
            if key in header:
                result[key] = header[key]
        return pack_packet(result, encoded.tobytes())

    async def worker(s):
        while not s.closing:
            await s.available.wait()
            s.available.clear()
            if s.pending is None:
                continue
            header, jpeg, received_at = s.pending
            s.pending = None
            if time.monotonic() - received_at > max_pending_age:
                s.dropped += 1
                continue
            try:
                # Shield prevents timeout/disconnect from pretending the model thread
                # stopped. Keep the session slot until that thread and close() finish.
                s.inflight = asyncio.create_task(asyncio.to_thread(process, s, header, jpeg, received_at))
                packet = await asyncio.wait_for(asyncio.shield(s.inflight), process_timeout)
                if s.closing or s.ws.closed:
                    return
                await send(s, packet, binary=True)
                s.sent += 1
                s.last_output = time.monotonic()
            except asyncio.TimeoutError:
                await fail(s, 'TIMEOUT', 'Processing or output send deadline exceeded', header['frame_id'])
                return
            except Exception as exc:
                LOG.exception('Frame failed stream=%s session=%s frame=%s', s.stream_id, s.session_id, header['frame_id'])
                await fail(s, 'INVALID_FRAME' if isinstance(exc, ProtocolError) else 'PROCESSING_FAILED',
                           str(exc), header['frame_id'])
                return

    async def dispose(s):
        try:
            if s.inflight:
                await asyncio.gather(s.inflight, return_exceptions=True)
            if s.processor and hasattr(s.processor, 'close'):
                await asyncio.to_thread(s.processor.close)
        except Exception:
            LOG.exception('Processor cleanup failed for %s', s.stream_id)
        finally:
            if sessions.get(s.stream_id) is s:
                del sessions[s.stream_id]
            LOG.info('Session closed %s', s.snapshot())

    async def connect(request):
        stream_id = request.match_info['stream_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', stream_id):
            return web.json_response({'error': 'INVALID_STREAM_ID'}, status=400)
        if stopping:
            return web.json_response({'error': 'SHUTTING_DOWN'}, status=503)
        if stream_id in sessions:
            return web.json_response({'error': 'STREAM_BUSY', 'retry_after_ms': 250}, status=409)
        if len(sessions) >= max_sessions:
            return web.json_response({'error': 'CAPACITY', 'retry_after_ms': 1000}, status=503)
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=MAX_PACKET, compress=False, timeout=2)
        s = Session(stream_id, ws)
        sessions[stream_id] = s  # Reserve before the first await, including handshake.
        try:
            await ws.prepare(request)
            await send(s, dict(type='ready', version=1, stream_id=stream_id,
                               session_id=s.session_id, processor=processor_name,
                               max_packet_bytes=MAX_PACKET, idle_timeout_s=idle_timeout))
            s.worker = asyncio.create_task(worker(s))
            while not ws.closed:
                try:
                    msg = await asyncio.wait_for(ws.receive(), idle_timeout)
                except asyncio.TimeoutError:
                    await fail(s, 'IDLE_TIMEOUT', 'No video frame received before deadline')
                    break
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
                try:
                    if msg.type != WSMsgType.BINARY:
                        raise ProtocolError('Expected binary frame packet')
                    header, jpeg = unpack_packet(msg.data)
                    validate_frame(header)
                    if header['frame_id'] <= s.last_id:
                        raise ProtocolError('frame_id must strictly increase within a session')
                except ProtocolError as exc:
                    await fail(s, 'INVALID_FRAME', str(exc))
                    break
                s.last_id = header['frame_id']
                s.received += 1
                if s.pending is not None:
                    s.dropped += 1
                s.pending = (header, jpeg, time.monotonic())
                s.available.set()
        finally:
            s.closing = True
            s.pending = None
            if s.worker:
                s.worker.cancel()
                await asyncio.gather(s.worker, return_exceptions=True)
            await ws.close()
            task = asyncio.create_task(dispose(s))
            disposal.add(task)
            task.add_done_callback(disposal.discard)
        return ws

    async def health(request):
        return web.json_response(dict(alive=True, ready=prepared and not stopping,
                                      model=getattr(factory, 'status', lambda: None)(),
                                      protocol_version=1, processor=processor_name,
                                      max_sessions=max_sessions, active_sessions=len(sessions)))

    async def status(request):
        return web.json_response({'streams': [s.snapshot() for s in sessions.values()]})

    async def shutdown(_app):
        nonlocal stopping
        stopping = True
        await asyncio.gather(*(s.ws.close(code=1001, message=b'Service shutdown')
                               for s in list(sessions.values())), return_exceptions=True)

    async def cleanup(_app):
        if disposal:
            await asyncio.gather(*list(disposal), return_exceptions=True)

    async def startup(_app):
        nonlocal prepared
        prepare = getattr(factory, 'prepare', None)
        if prepare is not None:
            await asyncio.to_thread(prepare)
        prepared = True

    app.on_startup.append(startup)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    app.add_routes([web.get('/health/live', health), web.get('/health/ready', health),
                    web.get('/v1/streams', status), web.get('/v1/streams/{stream_id}/ws', connect)])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--processor', default='service.stream_processor:create_processor')
    parser.add_argument('--max-sessions', type=int, default=4)
    parser.add_argument('--process-timeout', type=float, default=2.)
    parser.add_argument('--idle-timeout', type=float, default=15.)
    parser.add_argument('--jpeg-quality', type=int, default=85)
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or not 1 <= args.max_sessions <= 16
            or not 0 < args.process_timeout <= 60 or not 0 < args.idle_timeout <= 300
            or not 1 <= args.jpeg_quality <= 100):
        parser.error('Invalid port, session count, timeout or JPEG quality')
    module, name = args.processor.split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    token = os.environ.get('STREAM_API_TOKEN', '')
    if args.host not in ('127.0.0.1', '::1', 'localhost') and not token:
        parser.error('Non-loopback binding requires STREAM_API_TOKEN')
    logging.basicConfig(level=logging.INFO)
    web.run_app(make_app(factory, token=token, max_sessions=args.max_sessions,
                         process_timeout=args.process_timeout, idle_timeout=args.idle_timeout,
                         jpeg_quality=args.jpeg_quality, processor_name=args.processor),
                host=args.host, port=args.port, shutdown_timeout=5)


if __name__ == '__main__':
    main()
