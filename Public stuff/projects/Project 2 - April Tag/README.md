# Project 2 — AprilTag Parking

A LEGO Education Double Motor "car" finds a printed AprilTag with a
camera, turns to face it, drives up to it, and centers itself on it —
built up in stages, from a stream sanity-check to a full autonomous
search/align/drive pipeline.

## Files, in the order they were built

1. **`apriltag_stream_test.py`** — the very first step: just opens an
   iPhone's camera stream (via a free IP-camera app, MJPEG over HTTP)
   and shows it live with an FPS counter. No AprilTag detection, no
   motors. Get this rock-solid before anything else — AprilTag
   detection and motor control both depend on actually receiving
   frames reliably.

2. **`apriltag_center.py`** — the first real control loop. A phone
   (mounted sideways) streams to a plain LEGO Double Motor "car" (no
   rotating camera platform yet). The car drives straight
   forward/backward only — no turning — using a PD controller on the
   AprilTag's horizontal pixel offset, so it always ends up centered on
   the tag. This is where the PD gains got tuned, and where we hit our
   first real hardware bug: a derivative term reacting to ordinary
   AprilTag-detection pixel noise made the motor jerk violently even at
   a tiny proportional gain.

3. **`apriltag_center_webcam.py`** — the same PD-centering algorithm,
   camera and tag roles swapped: the AprilTag is taped to the car
   itself, and a stationary laptop webcam (via `camlib.pick_camera()`,
   not a phone stream) watches it drive back and forth. Used to
   confirm the controller worked independent of the phone-streaming
   setup.

4. **`apriltag_parking.py`** — the full project. An iPhone now sits on
   its own LEGO Single Motor rotating platform mounted on top of the
   Double Motor "car," so the camera can be aimed independently of the
   car's own heading. Every frame runs a 3-state pipeline:
   - **SEARCH** — a human manually spins the camera platform with a
     LEGO Controller's joystick to hunt for the tag; the car sits
     still.
   - **ALIGN** *(autonomous)* — the moment the tag is spotted, the
     platform keeps it centered in view (vision P-control) while the
     car turns in place to face the same direction, using the
     platform's own motor-rotation reading as its error signal — so
     the two motors counter-rotate against each other and the camera
     never loses the tag mid-turn. Which physical direction the car
     needs to turn isn't hardcoded: ALIGN runs a brief probe turn first
     and watches which way actually helps, instead of relying on a
     hand-guessed sign constant.
   - **DRIVE** *(autonomous)* — once the car is pointed straight at the
     tag, it drives forward and centers itself with the same P-control
     style as `apriltag_center.py`.

   Losing the tag at any point drops straight back to SEARCH — no
   guessing, no coasting on the last command.

## Setup

```bash
python -m venv my_env
my_env\Scripts\activate.bat        # Windows
source my_env/bin/activate         # macOS/Linux
pip install --upgrade pip
pip install legoeducation opencv-python numpy
```

- `lelib.py` / `camlib.py` (from `Public stuff/useful libraries/`) are
  picked up automatically via a `sys.path` fix at the top of each
  script — no need to copy them into this folder.
- For the phone-streaming scripts (`apriltag_stream_test.py`,
  `apriltag_center.py`, `apriltag_parking.py`): install a free
  IP-camera app on the phone, start its server, and paste its shown
  URL into `STREAM_URL` at the top of each script. Verify the stream
  first with `apriltag_stream_test.py` before running anything else.
- Print an AprilTag from the 36h11 family (`cv2.aruco.DICT_APRILTAG_36h11`).
- Fill in the `CARD_SERIAL`/`CARD_COLOR` placeholders in each script
  with the values on your LEGO Connection Card(s) — `apriltag_parking.py`
  connects the Double Motor, Single Motor, and Controller all through
  the *same* physical card.
- `apriltag_parking.py` needs the camera platform manually pointed
  dead-ahead (matching the car's own front) before you press run — see
  its docstring for why.

## Run

```bash
my_env/Scripts/python "Public stuff/projects/Project 2 - April Tag/apriltag_stream_test.py"
my_env/Scripts/python "Public stuff/projects/Project 2 - April Tag/apriltag_center.py"
my_env/Scripts/python "Public stuff/projects/Project 2 - April Tag/apriltag_center_webcam.py"
my_env/Scripts/python "Public stuff/projects/Project 2 - April Tag/apriltag_parking.py"
```

## Biggest failures along the way

- **The camera feed would stutter every time a motor command was
  sent.** Every `legoeducation` motor command blocks the thread that
  issues it until the BLE write completes — even ones marked
  `blocking=False` (that flag only skips waiting for a completion
  acknowledgment, not the send itself). Since motor commands and
  `cap.read()`/`cv2.imshow()` were running on the same thread, driving
  the car was stalling the vision loop. Fixed by moving all motor
  commands onto their own dedicated background thread.
- **A derivative (D) term amplified ordinary vision noise into violent
  motor commands.** A few pixels of normal AprilTag-detection jitter,
  divided by a small time step, produced a huge derivative spike —
  snapping the motor to full speed almost every frame even with a tiny
  proportional gain. Fixed with a lower derivative gain and a
  firmware-level acceleration/deceleration ramp (`motor_set_acceleration`)
  so speed changes are physically smoothed regardless of how spiky the
  computed command is.
- **The car's turn direction during ALIGN wasn't just "backward
  sometimes" — it could run away.** Driving the car's turn off a raw,
  unbounded motor-rotation counter (instead of wrapping it to the
  shortest angle) meant a platform that had been spun around more than
  once while searching would make the car try to unwind several extra
  full turns instead of just turning the short way — looking exactly
  like it "lost track of" where the tag was.
