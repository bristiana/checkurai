"""Raw ID check for both arms (no calibration needed, torque untouched). Move ONE joint at a time. ENTER to finish."""
import serial, time, sys, select
ARMS = {"LEADER": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00",
        "FOLLOWER": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"}
NAMES = ["", "1 base (shoulder_pan)", "2 shoulder_lift", "3 elbow_flex", "4 wrist_flex", "5 wrist_roll", "6 gripper"]
def pkt(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
def pos(ser, sid):
    ser.reset_input_buffer(); ser.write(pkt(sid, 2, bytes([56, 2]))); ser.flush()
    t0 = time.time(); buf = b""
    while time.time() - t0 < 0.03:
        buf += ser.read(64); i = buf.find(b"\xff\xff")
        if i >= 0 and len(buf) >= i + 8:
            v = buf[i+5] | (buf[i+6] << 8); return -(v & 0x7FFF) if v & 0x8000 else v
    return None
sers = {k: serial.Serial(p, 1_000_000, timeout=0.003) for k, p in ARMS.items()}
hist = {(k, i): [] for k in ARMS for i in range(1, 7)}
while True:
    out = ["\033[H\033[J=== IDENTIFY: move ONE joint at a time; its row should light up ===", ""]
    for k, s in sers.items():
        out.append(f"--- {k} ---")
        for i in range(1, 7):
            p = pos(s, i); h = hist[(k, i)]; h.append(p if p is not None else 0); del h[:-8]
            moving = p is not None and (max(h) - min(h)) > 25
            out.append(f"  {NAMES[i]:22s} {str(p):>6s}  {'<<< MOVING' if moving else ''}")
        out.append("")
    out.append("Press ENTER when done.")
    print("\n".join(out), flush=True)
    if select.select([sys.stdin], [], [], 0.1)[0]:
        sys.stdin.readline(); break
