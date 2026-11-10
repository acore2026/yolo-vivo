"""Hospital robot car HTTP/WebRTC service."""
import argparse
import asyncio
import os
from pathlib import Path
import uuid
import time

import cv2

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
import yaml

from .video_track import OutputTrack
from .car_pipeline import CarPipeline


def make_app(pipe, token='', ice_urls=()):
    @web.middleware
    async def auth(request, handler):
        if request.path != '/' and token and request.headers.get('Authorization') != 'Bearer '+token:
            raise web.HTTPUnauthorized()
        return await handler(request)
    app = web.Application(middlewares=[auth],client_max_size=128*1024)
    peers, sockets, tasks = {}, set(), set()
    lock = asyncio.Lock()

    async def index(request):
        return web.FileResponse(Path(__file__).with_name('car_viewer.html'))

    async def status(request):
        return web.json_response(pipe.snapshot())

    async def live(request):
        return web.json_response({'alive':True})

    async def ready(request):
        fresh = pipe.detection()['fresh']
        return web.json_response({'ready':fresh},status=200 if fresh else 503)

    async def control(request):
        async with lock:
            try:
                await asyncio.to_thread(pipe.start if request.match_info['action']=='start' else pipe.stop)
            except Exception:
                raise web.HTTPServiceUnavailable(text='Startup failed; see monitor/status')
        return await status(request)

    async def detections(request):
        return web.json_response(pipe.detection())

    async def frame(request):
        item = pipe.rendered
        if (item is None or time.monotonic() - item[1] >= .5
                or pipe.status in ('STOPPED', 'STOPPING', 'ERROR')):
            return web.json_response({'error': 'No fresh video', **pipe.snapshot()},
                                     status=503, headers={'Cache-Control': 'no-store'})
        ok, encoded = await asyncio.to_thread(
            cv2.imencode, '.jpg', item[2], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise web.HTTPInternalServerError(text='JPEG encoding failed')
        return web.Response(body=encoded.tobytes(), content_type='image/jpeg',
                            headers={'Cache-Control': 'no-store', 'X-Frame-Id': str(item[0])})

    async def stream(request):
        if len(sockets)>=8:
            raise web.HTTPServiceUnavailable(text='Viewer limit reached')
        response = web.StreamResponse(headers={'Content-Type':'text/event-stream','Cache-Control':'no-cache','X-Accel-Buffering':'no'})
        await response.prepare(request)
        task = asyncio.current_task()
        sockets.add(response)
        tasks.add(task)
        import json
        last = None
        try:
            while True:
                item = pipe.detection()
                key = (item.get('frame_id'),item['fresh'])
                if key != last:
                    await asyncio.wait_for(response.write(('data: '+json.dumps(item)+'\n\n').encode()),5)
                    last = key
                await asyncio.sleep(.03)
        except (ConnectionError,asyncio.TimeoutError):
            pass
        finally:
            sockets.discard(response)
            tasks.discard(task)
        return response

    async def offer(request):
        if len(peers)>=4:
            raise web.HTTPServiceUnavailable(text='Viewer limit reached')
        body = await request.json()
        if not isinstance(body,dict) or body.get('type')!='offer' or not isinstance(body.get('sdp'),str):
            raise web.HTTPBadRequest(text='Expected SDP offer')
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(urls=u) for u in ice_urls]))
        pid = str(uuid.uuid4())
        track = OutputTrack(pipe)
        peers[pid] = pc
        async def close():
            peers.pop(pid,None)
            track.stop()
            await pc.close()
        @pc.on('connectionstatechange')
        async def changed():
            if pc.connectionState in ('failed','closed'):
                await close()
        async def expire():
            await asyncio.sleep(30)
            if pc.connectionState!='connected':
                await close()
        timer = asyncio.create_task(expire())
        tasks.add(timer)
        timer.add_done_callback(tasks.discard)
        try:
            await pc.setRemoteDescription(RTCSessionDescription(**body))
            pc.addTrack(track)
            await pc.setLocalDescription(await pc.createAnswer())
            return web.json_response(dict(type=pc.localDescription.type,sdp=pc.localDescription.sdp,peer_id=pid))
        except Exception:
            await close()
            raise web.HTTPBadRequest(text='Invalid SDP')

    async def delete(request):
        pc = peers.get(request.match_info['peer_id'])
        if pc:
            await pc.close()
        return web.json_response({'closed':True})

    async def cleanup(app):
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks),return_exceptions=True)
        await asyncio.gather(*(pc.close() for pc in list(peers.values())),return_exceptions=True)
        await asyncio.to_thread(pipe.stop)
    app.on_shutdown.append(cleanup)
    app.router.add_get('/v1/video/frame.jpg', frame)
    app.add_routes([web.get('/',index),web.get('/health/live',live),web.get('/health/ready',ready),web.get('/v1/monitor/status',status),web.post('/v1/monitor/{action:start|stop}',control),web.get('/v1/detections',detections),web.get('/v1/detections/stream',stream),web.post('/v1/webrtc/offer',offer),web.delete('/v1/webrtc/{peer_id}',delete)])
    return app


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('car_config.yml'))
    for key in ('model','source','device','host','camera-id'):
        parser.add_argument('--'+key,default=argparse.SUPPRESS)
    parser.add_argument('--port',type=int,default=argparse.SUPPRESS)
    parser.add_argument('--confidence',type=float,default=argparse.SUPPRESS)
    args = vars(parser.parse_args())
    config = yaml.safe_load(args.pop('config').read_text())
    config.update(args)
    if not 0<=config['confidence']<=1 or not 1<=config['port']<=65535:
        parser.error('Invalid confidence/port')
    for key in ('model','source'):
        value = config[key]
        if '://' not in value:
            config[key] = str(root/Path(value).expanduser())
    pipe = CarPipeline(config['model'],config['source'],str(config['device']),config['confidence'],config['camera_id'])
    web.run_app(make_app(pipe,os.environ.get('CAR_API_TOKEN',''),config.get('ice_url',[])),host=config['host'],port=config['port'])


if __name__=='__main__':
    main()
