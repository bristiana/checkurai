# Conveyor belt for the Intel cell (plan, 13 Nov 23:55)

**Goal:** a 40–50 cm 3D-printed belt that brings Duplo bricks to the SO-101.
- The laptop controls the belt in **indexed** mode: the belt runs until a brick has arrived, then stops; the arm picks while the belt is stopped and never picks from a moving belt.
- **Constraints:** venue parts only (no shopping). A teammate builds it starting tonight.

## Why indexed, never moving
With LeRobot's ACT defaults (chunk_size = n_action_steps = 100, at 30 fps), the policy plays ~3.3 s of actions without looking again. A moving brick would be gone by the time the gripper closes.

## Design: 3Core "Conveyor Generator" (MakerWorld)
- **Links:** https://makerworld.com/en/models/2188718-conveyor-generator · assembly guide (V2): https://makerworld.com/en/models/1980909
- **Checked on the page (23:50):**
  - Module length 140–350 mm, total width 55–150 mm, belt width = total width − 14 mm.
  - Height is adjustable, and modules join with printed connectors.
  - The belt is made of printed hinged slats joined with pieces of filament, driven by sprockets: **no belt material, bearings or rods to buy.**
  - Drive-side options: Bambu TT motor (5 V, 59 RPM), continuous-rotation servo (micro-servo size), or a hand crank.
  - The default profile (220 × 74 mm module, 20 mm legs) is **14.7 h on 7 plates (P1S)**, with a 4.8★ rating.
  - The generator outputs one assembled 3MF. In the slicer: Split to Objects → arrange the parts on several plates → print.
- **Settings for us:**
  - **2 modules × 200 mm (40 cm belt).** Use 2 × 225 mm only if the table is deeper than ~60 cm (see layout).
  - **Total width 100 mm** (86 mm belt; a 2×4 fits crosswise). If printers are scarce, use the default 74 mm (60 mm belt; 2×4s ride lengthwise).
  - **Low legs** (~20 mm).
  - **Include the end chute**: it drops bricks into the pick tray.
- **Print:** about 30–38 h of printer time and 0.7–1 kg PLA. Spread across 7–8 printers that's about 4–5 h wall time, so **start by ~01:00** to have parts by ~06:00. Assembly (pinning the slat links) is about 1.5–2.5 h.
- **Printers (desk page, 23:29):** 10 free and the queue is empty: B1, B4 Core One; B2, B3, F3 P1S; F4 P2S; B5, B6 MK4; F5, F6 Core One+. B7, F1 and F2 are out of service.
  - Keep every part within **240 × 200 mm** so it fits all of them.
  - Filament stock is unknown, so ask the desk.
- **Licence:** personal use, no redistribution. **Never commit the 3MF/STL files to the repo.**
- **Generating the file needs a MakerWorld (Bambu) login** and Bambu Studio or PrusaSlicer on someone's laptop.

## DECISION (Marius, 14 Nov ~00:10): full 40 cm, Atech drive
- **Build:** 2 × 200 mm, 100 mm wide, low legs, end chute plus a ~12×12 cm tray.
- **Drive:** the Atech 14-port board (Frame 2.2, ESP32) with a DC Motor module. Marius has the board and 2 motor modules.
  - SDK: `~/atech-venv` on the Intel laptop. Firmware: `~/checkurai/belt/project.yaml`.
  - Actions: `dc_motor_speed` (−255..255), `dc_motor_stop`, `dc_motor_brake`.
- **Power (from the SDK: dc_motor.h, docs/guide.md):** the motor runs on the board's "12 V" rail. Only the **Power Jack** or **USB-C PD** module feeds that rail; the standard USB-C module doesn't, so **USB alone won't spin it**. The rail is unregulated, max 12 V, and the motor's rating is unknown, so ask Atech. Power modules need no `project.yaml` entry, and their port must stay empty.
- **Motor geometry (Atech CAD bounds, `dc_motor/sim.yaml`):**
  - The module is a 19.5 mm cell, with the motor housing (19.5 × 20.1 × 19.1 mm) sticking out of the port.
  - Shaft: **Ø3 mm × 8.8 mm**, ~10 mm above the board, parallel to it.
  - So the motor must sit in an **edge port**, and the board sits beside the drive end on a printed cradle.
- **Coupler plan:** generate the conveyor with the **TT-motor drive side**. A printed coupler has a TT-style double-D male end (Ø5.4 mm, 3.7 mm flats) that plugs into the sprocket, and a Ø3 mm bore for the Atech shaft. The generator parts stay unchanged.
- **Timeline (PLAN.md D16):**
  - The belt should be fixed in place by ~08:30. Demos never wait past 09:30: without the belt by then, they're recorded without it.
  - The pick policy is trained on table zones too.
  - 3MF/STL files are in .gitignore.

## Drive options researched (before the decision)
1. **Spare STS3215 + its own Waveshare bus adapter + its own power supply** (ask the hardware desk / Intel).
   - The laptop already drives these with LeRobot: velocity mode = `Operating_Mode` 1, `Goal_Velocity` (addr 46, raw steps/s, sign bit 15). The LeKiwi robot does exactly this.
   - Gives ~45–52 rpm, so a ~25 mm sprocket gives ~6 cm/s. That's closed-loop, with plenty of torque.
   - Needs a **printed bracket + horn-to-shaft coupler**, because the generator's servo side fits micro servos only. Claude can design these once the servo is confirmed.
   - Plug the adapter **directly into the laptop, not the hub**. Supply voltage must match the servo variant (7.4 V C001 → 5–7.4 V; 12 V C018 → 12 V).
   ```python
   bus = FeetechMotorsBus(port, {"belt": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)})
   bus.connect(); bus.disable_torque()
   bus.write("Operating_Mode", "belt", OperatingMode.VELOCITY.value); bus.enable_torque()
   bus.write("Goal_Velocity", "belt", 2000)   # 0 = stop
   ```
2. **Atech (atech.dev):** all 9 dev kits were checked out, but "Atech motherboard + modules" had 17 of 20 free (contents not listed).
   - Needs the **DC Motor module** (DRV8833, speed −255…255), the **Distance module** (VL53L5CX ToF) and the **Power Jack module** with a 5–9 V supply. The standard USB-C module doesn't power the motor rail.
   - Laptop control: USB serial, `pip install atech`, `board.call("belt_speed", 160)`.
   - The motor is ~20 mm with a 3 mm shaft and its gearing isn't published, so the pulley adapter must be measured and printed.
3. **Last resort: Arduino UNO Q.**
   - Its USB-C goes to the Linux side only. It needs a separate motor driver (DRV8833/TB6612) and a motor, neither of which is at the desk, plus Bridge/App Lab or a USB-serial adapter on D0/D1.
   - Too slow to set up this weekend.
- **No motor at all:** the generator's **hand crank** (36 min print) keeps the belt as a demo prop. Not autonomous.

## "Brick has arrived" sensing
Use the **RealSense depth** over the pick-tray region: no extra hardware, and it feeds the 3D twin. Run the belt until a new brick appears in the tray, stop, then pick. Add a 15 s timeout ("line empty").

## Layout (from Marius's top-down photo; dimensions are estimates)
- **Estimates (MEASURE before printing):** table ~50 cm deep (a beer table). Duplo scale suggests ~1 px/mm in the photo.
- **Setup:** operator edge = the side where the arms' bases and the Intel laptop are. The RealSense tripod is at the left end, the bowls are right of the arm.
- **Proposal:** the belt runs **across the table depth, ~15–20 cm left of the follower's base**, from the far edge (infeed) toward the operator edge. The end chute drops bricks into a **~12 × 12 cm pick tray** (tape plus a low printed fence) at the front-left of the arm.
  - Belt 40 cm + tray ~10 cm = the whole ~50 cm depth.
  - **Move the closed Lenovo laptop, phone, bottles, USB hub and the inspection webcam** out of that strip. The webcam currently sits right where the belt would go.
- **Before printing:** tape the footprint and check with the **leader arm in teleop** that the gripper reaches the whole tray. Check the RealSense image too: the belt end and the tray must be in frame.
```
            far edge  (infeed: drop bricks here)
  ┌───────────────────────────────────────────────────────┐
  │ RealSense      ║▓▓║                                    │
  │ tripod         ║▓▓║  belt 40 cm                (bowls) │
  │                ║▓▓║  runs toward operator              │
  │                ║▓▓║                                    │
  │                [tray] ← pick here    [FOLLOWER]  ... [LEADER]
  └───────────────────────────────────────────────────────┘
            operator edge (Intel laptop)
```

## Judging fit
- **Task success (30%):**
  - Bricks drop off the chute at random spots and angles, so position robustness still gets tested.
  - Running the belt for several bricks piles them in the tray, which gives **clutter on demand**.
  - Bricks a judge places on the tray by hand go through the **same pick policy**, so the judges' own tests still work.
  - **Risk:** a fixed V-pocket end stop would remove the variation, so don't build one.
- **Dataset (20%):** the belt must be in place **before the real demos**. Tonight's smoke-test demos are throwaway anyway.
- **Intel integration (30%):** the belt is part of the laptop's state machine, and the depth "arrived" check feeds the 3D twin. Report throughput (bricks/min) as a production KPI.
- **Efficiency (20%):** neutral, since the belt is stopped while the arm works.

## Failure modes
- Moving the belt after the real demos invalidates the policy. **Fix its position with tape / double-sided tape.**
- Parts that don't fit (reported in comments), slat pins popping out, the chain jumping a sprocket.
- No spare servo or motor at the desk. Then use Atech, or the hand crank as a prop.
- The desk won't run 7–8 printers in parallel for one team. Then use 1 module (22 cm), or the 74 mm width.
- Two bricks drop at once, a brick bounces out of the tray, or a brick sticks on the chute.
- The belt frame blocks a camera view, or the gripper hits the tray fence.
- A new USB device on the flaky hub. Plug it into the laptop instead.

## Minimum version (3–4 h)
**One default module** (220 × 74 mm, the designer's ready 7-plate P1S profile, no re-slicing):
- 7 plates on 7 printers ≈ 2–2.5 h, plus ~1 h assembly.
- Drive: STS3215 + coupler, or the hand crank.
- 22 cm long, so it looks less impressive, but it's the same control loop.

## Tonight, in order
1. **Hardware desk, now:**
   - Is there a spare **STS3215 + Waveshare bus servo adapter + power supply**? Otherwise, does the "Atech motherboard + modules" bag have a **DC Motor + Distance + Power Jack** module and a 5–9 V supply?
   - Which **filament** is there, and can we use **7–8 printers in parallel overnight**?
   - Any M2/M3 screws?
2. **Intel (Rajesh / Marvin, Discord or 5th floor):** is an extra fixture like a conveyor OK, and how will judges place bricks during evaluation?
3. Measure the table depth, tape out the layout, and check reach with the leader arm.
4. Generate the 3MF on MakerWorld (settings above), split to plates and queue the prints by ~01:00.
5. Sat ~06:00–08:30: assemble, mount the drive, run the belt test script. **Record the real demos only after the belt is fixed in place.**
