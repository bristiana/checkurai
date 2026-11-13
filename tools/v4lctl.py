"""List / set V4L2 controls without v4l2-ctl. Usage: v4lctl.py /dev/videoN [name=value ...]"""
import fcntl, os, struct, sys
VIDIOC_QUERYCTRL, VIDIOC_G_CTRL, VIDIOC_S_CTRL = 0xC0445624, 0xC008561B, 0xC008561C
NEXT = 0x80000000
fd = os.open(sys.argv[1], os.O_RDWR)
ctrls, cid = {}, NEXT
while True:
    buf = bytearray(struct.pack("II32siiiiIII", cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
    try: fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf)
    except OSError: break
    i, typ, name, mn, mx, step, dflt, flags = struct.unpack("II32siiiiI", bytes(buf[:60]))
    name = name.split(b"\0")[0].decode(); cid = i | NEXT
    if typ == 6: continue  # control class header
    g = bytearray(struct.pack("Ii", i, 0))
    try: fcntl.ioctl(fd, VIDIOC_G_CTRL, g); val = struct.unpack("Ii", g)[1]
    except OSError: val = None
    ctrls[name.lower().replace(" ", "_").replace(",", "")] = i
    print(f"{name:40s} id=0x{i:08x} min={mn} max={mx} step={step} default={dflt} value={val} flags=0x{flags:x}")
for kv in sys.argv[2:]:
    k, v = kv.split("="); s = bytearray(struct.pack("Ii", ctrls[k], int(v)))
    fcntl.ioctl(fd, VIDIOC_S_CTRL, s); print("set", k, "=", v)
