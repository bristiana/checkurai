from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00"
F = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7045345-if00"
leader = SOLeader(SOLeaderTeleopConfig(port=L, id="checkurai_leader"))
follower = SOFollower(SOFollowerRobotConfig(port=F, id="checkurai_follower"))
for name, dev in [("LEADER", leader), ("FOLLOWER", follower)]:
    dev.bus.connect()
    cal = dev.bus.read_calibration()
    print(name, "file-calibrated:", dev.is_calibrated if dev.calibration else "no file")
    for m, c in cal.items():
        print(f"  {m:14s} id={c.id} offset={c.homing_offset:6d} range=[{c.range_min:4d},{c.range_max:4d}]")
    dev.calibration = cal
    dev.bus.calibration = cal
    pos = dev.bus.sync_read("Present_Position")
    print("  pose:", {k: round(v, 1) for k, v in pos.items()})
    dev.bus.disconnect(disable_torque=False)
