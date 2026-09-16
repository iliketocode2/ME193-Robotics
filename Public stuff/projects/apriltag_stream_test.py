"""
Validate an iPhone camera stream before building the AprilTag-parking
project on top of it.

Setup:
    1. On the iPhone, install a free IP-camera app that streams over the
       local network -- e.g. "RTSP & RTMP Cam - OctoStream" (RTSP) or
       "SimpleIPCamera" (MJPEG over HTTP). Start its server; the app
       displays the exact URL to use -- copy it exactly, don't guess.
    2. Make sure the iPhone and this computer are on the SAME Wi-Fi
       network.
    3. Paste that URL into STREAM_URL below.

Run:
    my_env/Scripts/python "Public stuff/projects/apriltag_stream_test.py"

This does nothing but open the stream and show it live with an FPS
counter -- get this working reliably before anything else in this
project, since AprilTag detection and motor control both depend on
actually receiving frames with low, stable latency.
"""
import time

import cv2

# --- Fill this in from your streaming app's own display ----------------
STREAM_URL = "http://10.243.67.39:8080/stream.jpeg"  # <-- replace with your app's shown URL

# Some streaming apps send the camera's native sensor orientation
# without correcting for how the phone is physically mounted, so a
# vertically-mounted phone can produce a frame that's rotated 90
# degrees from upright. None = no rotation; otherwise one of
# cv2.ROTATE_90_CLOCKWISE / cv2.ROTATE_90_COUNTERCLOCKWISE / cv2.ROTATE_180.
# Verify by panning the phone left/right in front of it: the image
# should pan left/right too, not up/down -- if it moves the wrong axis,
# try the other 90-degree constant, not just "the picture looks upright".
FRAME_ROTATION = None  # phone remounted landscape -- re-verify with the pan test if this changes again


def main():
    cap = cv2.VideoCapture(STREAM_URL)
    # Keep the buffer as small as possible -- RTSP/MJPEG sources otherwise
    # tend to accumulate latency over time (you'd be watching older and
    # older frames), which will matter a lot once this drives a live
    # control loop. Not every backend honors this, but it doesn't hurt.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open stream at {STREAM_URL!r}. Check: the phone "
            "and this computer are on the same Wi-Fi network, the "
            "streaming app's server is actually running and showing "
            "'connected'/active, and the URL was copied exactly as shown "
            "on the phone (not retyped)."
        )

    print("Stream opened. Press q in the video window to quit.")
    frame_count = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed -- stream may have dropped or the app was closed.")
                break

            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)

            frame_count += 1
            elapsed = time.time() - start
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame, f"{fps:.1f} fps  {frame.shape[1]}x{frame.shape[0]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imshow("iPhone stream test -- press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
