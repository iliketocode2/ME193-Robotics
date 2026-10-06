"""
Make the printable opponent cards: one tag36h11 AprilTag per opponent,
labeled with the opponent's name, saved to tags/. Print them (any size; the
white border around each tag matters -- don't crop it).

Run:
    my_env/Scripts/python "Public stuff/projects/Project 5 - Virtual Ping Pong/make_tags.py"
"""

import os

import cv2
import numpy as np

from opponents import OPPONENTS
from vision import APRILTAG_DICTIONARY

TAG_PX = 600
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tags")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
    for opp in OPPONENTS.values():
        tag = cv2.aruco.generateImageMarker(dictionary, opp["tag_id"], TAG_PX)
        margin = TAG_PX // 6
        card = np.full((TAG_PX + 2 * margin + 140, TAG_PX + 2 * margin), 255, np.uint8)
        card[margin:margin + TAG_PX, margin:margin + TAG_PX] = tag
        for i, (text, scale) in enumerate([(opp["name"], 1.6), (f'{opp["level"]}  -  tag36h11 #{opp["tag_id"]}', 1.0)]):
            (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
            y = TAG_PX + 2 * margin + 30 + i * 60
            cv2.putText(card, text, ((card.shape[1] - w) // 2, y), cv2.FONT_HERSHEY_DUPLEX, scale, 0, 2, cv2.LINE_AA)
        path = os.path.join(OUT_DIR, f'tag{opp["tag_id"]}_{opp["key"]}.png')
        cv2.imwrite(path, card)
        print("wrote", path)


if __name__ == "__main__":
    main()
