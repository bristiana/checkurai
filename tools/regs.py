import serial, sys, time
ser = serial.Serial(sys.argv[1], 1_000_000, timeout=0.005)
def pkt(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
def rd(sid, addr, n):
    ser.reset_input_buffer(); ser.write(pkt(sid, 2, bytes([addr, n]))); ser.flush()
    t0 = time.time(); buf = b""
    while time.time() - t0 < 0.05:
        buf += ser.read(64); i = buf.find(b"\xff\xff")
        if i >= 0 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i+3]:
            return buf[i+5:i+5+n]
    return None
def r16(sid, a):
    d = rd(sid, a, 2); return None if not d else d[0] | (d[1] << 8)
for sid in range(1, 7):
    m = rd(sid, 33, 1)
    print(f"id {sid}: minlim={r16(sid,9)} maxlim={r16(sid,11)} homing_ofs_raw={r16(sid,31)} mode={m[0] if m else None} pos_raw={r16(sid,56)}")
