"""Version 1: uint32 big-endian JSON length, UTF-8 JSON, JPEG bytes."""
import json
import struct

MAX_HEADER = 16 * 1024
MAX_JPEG = 4 * 1024 * 1024
MAX_PACKET = 4 + MAX_HEADER + MAX_JPEG
MAX_PIXELS = 3840 * 2160


class ProtocolError(ValueError):
    pass


def pack_packet(header, jpeg):
    data = json.dumps(header, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')
    if not 0 < len(data) <= MAX_HEADER or not 0 < len(jpeg) <= MAX_JPEG:
        raise ProtocolError('Header or JPEG exceeds packet limit')
    return struct.pack('!I', len(data)) + data + jpeg


def unpack_packet(data):
    if len(data) < 6 or len(data) > MAX_PACKET:
        raise ProtocolError('Invalid packet length')
    length = struct.unpack('!I', data[:4])[0]
    if not 0 < length <= MAX_HEADER or 4 + length >= len(data):
        raise ProtocolError('Invalid JSON header length')
    try:
        header = json.loads(data[4:4 + length],
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError('Invalid JSON header') from exc
    jpeg = data[4 + length:]
    if not isinstance(header, dict) or len(jpeg) > MAX_JPEG:
        raise ProtocolError('Invalid header object or JPEG length')
    return header, jpeg


def validate_frame(header):
    if type(header.get('version')) is not int or header['version'] != 1 or header.get('type') != 'frame':
        raise ProtocolError('Expected version=1, type=frame')
    fid = header.get('frame_id')
    if type(fid) is not int or not 0 <= fid <= 2**53 - 1:
        raise ProtocolError('frame_id must be an integer between 0 and 2^53-1')
    if header.get('codec') != 'jpeg':
        raise ProtocolError('Only codec=jpeg is supported')
    pts, base = header.get('pts'), header.get('time_base')
    if (pts is None) != (base is None):
        raise ProtocolError('pts and time_base must be provided together')
    if pts is not None and (type(pts) is not int or not -(2**63) <= pts < 2**63
                           or not isinstance(base, list) or len(base) != 2
                           or any(type(x) is not int or not 0 < x <= 2**31 - 1 for x in base)):
        raise ProtocolError('Invalid pts/time_base')


def decode_jpeg(jpeg):
    # Bound decoded allocation before passing compressed data to OpenCV.
    from io import BytesIO
    from PIL import Image
    import cv2
    import numpy as np
    try:
        with Image.open(BytesIO(jpeg)) as probe:
            w, h = probe.size
            if probe.format != 'JPEG' or not (0 < w <= 4096 and 0 < h <= 4096 and w * h <= MAX_PIXELS):
                raise ProtocolError('Expected JPEG up to 4096 per side and 3840x2160 pixels')
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError('Invalid JPEG') from exc
    if frame is None or frame.shape != (h, w, 3):
        raise ProtocolError('JPEG decode failed or dimensions changed')
    return frame
