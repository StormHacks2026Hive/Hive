import struct
from .models import ResultHeader

MAX_FRAME = 1_053_000

def decode_result(frame: bytes):
    if not 4 <= len(frame) <= MAX_FRAME:
        raise ValueError('Result frame is oversized or truncated')
    length = struct.unpack_from('<I', frame)[0]
    if not 1 <= length <= 4096 or 4 + length > len(frame):
        raise ValueError('Invalid binary header length')
    header = ResultHeader.model_validate_json(frame[4:4+length])
    payload = frame[4+length:]
    if len(payload) != header.byte_length:
        raise ValueError('Binary payload length does not match header')
    return header, payload
