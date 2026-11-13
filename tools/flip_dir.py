"""Flip a LEADER servo's direction via the Feetech 'Phase' register (bit0 = drive dir, bit7 = feedback dir).
Usage: flip_dir.py <motor> [--revert]. Saves the old values to ~/checkurai/logs/flip_<motor>.json. Leader stays limp."""
import json, os, statistics, sys, time
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
L = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B90148356-if00"
m = sys.argv[1]; LOG = os.path.expanduser(f"~/checkurai/logs/flip_{m}.json")
leader = SOLeader(SOLeaderTeleopConfig(port=L, id="checkurai_leader")); leader.connect(calibrate=False)
bus = leader.bus; bus.disable_torque()                       # torque off + EEPROM unlocked
rd = lambda r: int(bus.read(r, m, normalize=False))
pos = lambda: int(statistics.median(rd("Present_Position") for _ in range(7)))
try:
    if "--revert" in sys.argv:
        old = json.load(open(LOG))
        bus.write("Phase", m, old["phase"], normalize=False); time.sleep(0.2)
        bus.write("Homing_Offset", m, old["homing_offset"], normalize=False); time.sleep(0.2)
        print("reverted:", {"phase": rd("Phase"), "homing_offset": rd("Homing_Offset"), "pos": pos()})
    else:
        old = {"phase": rd("Phase"), "homing_offset": rd("Homing_Offset"), "pos": pos(),
               "min": rd("Min_Position_Limit"), "max": rd("Max_Position_Limit")}
        json.dump(old, open(LOG, "w")); print("before:", old)
        bus.write("Homing_Offset", m, 0, normalize=False); time.sleep(0.2)
        p0 = pos(); print("raw encoder (offset 0):", p0)
        new_phase = old["phase"] | 0x81
        bus.write("Phase", m, new_phase, normalize=False); time.sleep(0.3)
        p1 = pos(); ph = rd("Phase")
        print(f"phase now {ph} (wanted {new_phase}); raw after flip: {p1}; p0+p1 = {p0 + p1} (mirror -> ~4095)")
finally:
    leader.disconnect()
