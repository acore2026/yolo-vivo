"""Real HTTP, SSE and WebRTC smoke test using a separately supplied video."""
import asyncio
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aiohttp import web, ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from service.car_app import make_app
from service.car_pipeline import CarPipeline


async def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    pipe=CarPipeline('models/robot_car.pt',a.source,a.device)
    runner=web.AppRunner(make_app(pipe,token='test'))
    await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0)
    await site.start()
    port=site._server.sockets[0].getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    pc=RTCPeerConnection(RTCConfiguration(iceServers=[]))
    received=[]
    receiver_tasks=[]
    @pc.on('track')
    def track(t):
        async def receive():
            for _ in range(60):
                frame=await t.recv()
                assert (frame.width,frame.height)==(640,480)
                received.append(time.monotonic())
        receiver_tasks.append(asyncio.create_task(receive()))
    try:
        async with ClientSession() as anon:
            async with anon.get(base+'/v1/detections') as r:
                assert r.status==401
        async with ClientSession(headers={'Authorization':'Bearer test'}) as s:
            async with s.get(base+'/health/ready') as r:
                assert r.status==503
            async with s.post(base+'/v1/monitor/start') as r:
                assert r.status==200,await r.text()
            for _ in range(100):
                async with s.get(base+'/v1/detections') as r:
                    d=await r.json()
                if d['fresh']:
                    break
                await asyncio.sleep(.1)
            assert d['fresh'] and d['detections'],d
            async with s.get(base+'/v1/detections/stream') as r:
                line=await asyncio.wait_for(r.content.readline(),5)
                assert json.loads(line.decode().removeprefix('data: '))['fresh']
            pc.addTransceiver('video',direction='recvonly')
            await pc.setLocalDescription(await pc.createOffer())
            async with s.post(base+'/v1/webrtc/offer',json={'type':pc.localDescription.type,'sdp':pc.localDescription.sdp}) as r:
                assert r.status==200,await r.text()
                answer=await r.json()
            await pc.setRemoteDescription(RTCSessionDescription(type=answer['type'],sdp=answer['sdp']))
            for _ in range(150):
                if len(received)>=60:
                    break
                await asyncio.sleep(.1)
            assert len(received)==60,len(received)
            async with s.post(base+'/v1/monitor/stop') as r:
                assert (await r.json())['status']=='STOPPED'
            async with s.get(base+'/v1/detections') as r:
                stale=await r.json()
                assert not stale['fresh'] and stale['detected'] is None and stale['detections']==[]
            async with s.get(base+'/health/ready') as r:
                assert r.status==503
            print(json.dumps({'passed':True,'webrtc_frames':len(received),'width':640,'height':480,'receive_fps':59/(received[-1]-received[0]),'processed':pipe.processed,'dropped':pipe.dropped,'http_sse_auth_stop_checked':True},indent=2))
    finally:
        for t in receiver_tasks:
            t.cancel()
        await asyncio.gather(*receiver_tasks,return_exceptions=True)
        await pc.close()
        await runner.cleanup()


if __name__=='__main__':
    asyncio.run(main())
