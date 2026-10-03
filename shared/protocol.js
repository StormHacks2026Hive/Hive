// One binary frame = little-endian header size + UTF-8 JSON + raw RGBA bytes.
export function resultFrame(header, bytes) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const frame = new Uint8Array(4 + json.length + bytes.byteLength);
  new DataView(frame.buffer).setUint32(0, json.length, true);
  frame.set(json, 4);
  frame.set(bytes, 4 + json.length);
  return frame.buffer;
}
