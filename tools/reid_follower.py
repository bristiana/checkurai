"""Fix reversed servo IDs on the FOLLOWER: current 1..6 -> 6..1 (via temporary 11..16). Torque stays off."""
import serial, time, sys
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
ser = serial.Serial(PORT, 1_000_000, timeout=0.005)
def pkt(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
def xfer(sid, instr, params=b"", wait=0.05):
    ser.reset_input_buffer(); ser.write(pkt(sid, instr, params)); ser.flush()
    t0 = time.time(); buf = b""
    while time.time() - t0 < wait:
        buf += ser.read(64); i = buf.find(b"\xff\xff")
        if i >= 0 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i+3]:
            return buf[i:i+4+buf[i+3]]
    return None
def ping(sid): r = xfer(sid, 0x01); return bool(r and r[2] == sid)
def write(sid, addr, data): return xfer(sid, 0x03, bytes([addr]) + bytes(data))
def change_id(old, new):
    assert ping(old), f"id {old} not answering"; assert not ping(new), f"id {new} already taken"
    write(old, 40, [0])            # torque off
    write(old, 55, [0])            # unlock EEPROM
    write(old, 5, [new])           # new ID
    time.sleep(0.05)
    assert ping(new), f"id {new} not answering after change"
    write(new, 55, [1])            # lock EEPROM
    print(f"  {old} -> {new} ok")
before = [i for i in range(0, 21) if ping(i)]
print("before:", before)
if before != [1, 2, 3, 4, 5, 6]: sys.exit("unexpected IDs, aborting")
for i in range(1, 7): change_id(i, i + 10)
for i in range(11, 17): change_id(i, 7 - (i - 10))
print("after:", [i for i in range(0, 21) if ping(i)])
