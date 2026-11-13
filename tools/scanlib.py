"""Read-only Feetech STS3215 scan for SO-101 arms. Only PING and READ packets; never writes."""
import sys, time
import serial

BAUD = 1_000_000

def pkt(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

def xfer(ser, sid, instr, params=b""):
    ser.reset_input_buffer()
    ser.write(pkt(sid, instr, params))
    ser.flush()
    t0 = time.time()
    buf = b""
    while time.time() - t0 < 0.03:
        buf += ser.read(64)
        i = buf.find(b"\xff\xff")
        if i >= 0 and len(buf) >= i + 4:
            ln = buf[i + 3]
            if len(buf) >= i + 4 + ln:
                return buf[i:i + 4 + ln]
    return None

def read(ser, sid, addr, n):
    r = xfer(ser, sid, 0x02, bytes([addr, n]))
    if not r or r[2] != sid:
        return None
    data = r[5:5 + n]
    return data, r[4]  # data, error byte

for port in PORTS:
    print(f"=== {port} ===")
    try:
        ser = serial.Serial(port, BAUD, timeout=0.005)
    except Exception as e:
        print("  cannot open:", e)
        continue
