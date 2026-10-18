"""
make_tag.py -- generate the AprilTag that goes on top of OUR sumo robot.

tag36h11 id OUR_TAG_ID (config.py), same family the course uses everywhere
(Projects 2 and 5 detect it with cv2.aruco). Writes two files into tags/:

  sumo_tag36h11_id6.png        the tag + white quiet zone + labels (any size)
  sumo_tag36h11_id6_print.png  a 300 DPI letter page with the tag at exactly
                               TAG_PRINT_CM wide -- print at 100% / "actual size"

Mount it flat on the robot's top with the arrow under the tag pointing to the
robot's FRONT (the side that pushes; the color sensor is on the back).
Don't crop the white border -- the detector needs it.

Run:
    my_env\\Scripts\\python "Public stuff/projects/Project 6 - Sumo Wrestling/make_tag.py"
"""

import os

import cv2
import numpy as np

from config import APRILTAG_DICTIONARY, HERE, OUR_TAG_ID, TAG_PRINT_CM

DPI = 300
OUT_DIR = os.path.join(HERE, "tags")


def tag_card(tag_px):
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
    tag = cv2.aruco.generateImageMarker(dictionary, OUR_TAG_ID, tag_px)
    margin = tag_px // 5
    h = tag_px + 2 * margin + tag_px // 3
    card = np.full((h, tag_px + 2 * margin), 255, np.uint8)
    card[margin:margin + tag_px, margin:margin + tag_px] = tag
    cx = card.shape[1] // 2
    # "FRONT" arrow along the top quiet zone, pointing up = robot front
    a = margin // 2
    cv2.arrowedLine(card, (cx, margin - a // 3), (cx, a // 3), 0, max(2, tag_px // 120), tipLength=0.5)
    scale = tag_px / 600
    for i, text in enumerate([f"SUMO  tag36h11 #{OUR_TAG_ID}", "arrow = robot FRONT"]):
        s = scale * (1.1 if i == 0 else 0.8)
        (w, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, s, 2)
        y = margin + tag_px + int(margin * 0.9) + i * int(th * 1.8)
        cv2.putText(card, text, (cx - w // 2, y), cv2.FONT_HERSHEY_DUPLEX, s, 0, 2, cv2.LINE_AA)
    return card


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, f"sumo_tag36h11_id{OUR_TAG_ID}")

    cv2.imwrite(base + ".png", tag_card(600))

    # letter page at 300 DPI with the tag's black square exactly TAG_PRINT_CM wide
    page = np.full((int(11 * DPI), int(8.5 * DPI)), 255, np.uint8)
    card = tag_card(int(round(TAG_PRINT_CM / 2.54 * DPI)))
    y, x = DPI, (page.shape[1] - card.shape[1]) // 2
    page[y:y + card.shape[0], x:x + card.shape[1]] = card
    note = f"Print at 100% (actual size). Black square = {TAG_PRINT_CM:.1f} cm."
    cv2.putText(page, note, (DPI // 2, page.shape[0] - DPI // 2), cv2.FONT_HERSHEY_DUPLEX, 1.2, 0, 2, cv2.LINE_AA)
    cv2.imwrite(base + "_print.png", page, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # round-trip check: the detector must find exactly our id
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY),
                                  cv2.aruco.DetectorParameters())
    _, ids, _ = det.detectMarkers(cv2.imread(base + ".png", cv2.IMREAD_GRAYSCALE))
    assert ids is not None and list(ids.flatten()) == [OUR_TAG_ID], f"self-check failed: {ids}"
    print("wrote", base + ".png")
    print("wrote", base + "_print.png", "(detector self-check passed)")


if __name__ == "__main__":
    main()
