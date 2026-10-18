"""
songs_bridge.py -- the same win / lose songs as Project 3 (World Cup).

Laptop speaker: Project 3's songs.py plays waka_waka_final.mp3 (win) or
jb_sorry.mp3 (lose), loaded straight from the Project 3 folder so the mp3s
aren't duplicated in the repo.
Robot speaker: the double motor isn't connected to the laptop here, so we
publish T/song and the relay (unoq_app/python/relay.py) beeps the very same
SUCCESS_SONG / DEATH_SONG note lists on the double motor.
"""

import importlib.util
import json
import os
import threading

from config import HERE, TOPIC_SONG

_SONGS_PATH = os.path.join(HERE, "..", "Project 3 - World Cup", "songs.py")
_spec = importlib.util.spec_from_file_location("world_cup_songs", _SONGS_PATH)
songs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(songs)


def play(which, publish):
    """which = "win" | "lose". publish(topic, text) sends over MQTT."""
    publish(TOPIC_SONG, json.dumps({"song": which}))
    mp3 = songs.WIN_MP3 if which == "win" else songs.LOSE_MP3
    notes = songs.SUCCESS_SONG if which == "win" else songs.DEATH_SONG
    if not songs.play_computer_mp3(mp3):
        threading.Thread(target=songs.play_computer_song, args=(notes,), daemon=True).start()


def stop():
    songs.stop_computer_mp3()
