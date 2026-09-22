"""
mqtt_chat.py -- a minimal group chat over MQTT, using mqttlib.py.

Everyone who runs this script and points it at the same TOPIC/broker
sees everyone else's messages live, as chat bubbles: your own messages
on the right, everyone else's on the left. Type a message and press
Enter (or click Send) to publish it. See MQTTLIB.md for the underlying
pub/sub client this is built on.

Each message is published as a small JSON payload -- {"sender": ...,
"text": ...} -- instead of the raw string mqtt_test.py sends, so
everyone's client can tell who a message came from (including telling
your own messages apart from everyone else's, since the broker echoes
every publish back to every subscriber -- see MQTTLIB.md's "Reading
back your own publish" section).

Run:
    python "Public stuff/projects/mqtt_chat/mqtt_chat.py"
"""

import json
import os
import sys
import tkinter as tk
from tkinter import simpledialog

# mqttlib.py lives in the shared "useful libraries" folder, two levels
# up from this script (Public stuff/projects/mqtt_chat/) -- add it to
# sys.path so the import below resolves no matter where this script is
# run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "useful libraries"))

from mqttlib import MQTTClient

# Same topic as mqtt_test.py -- MQTT topics have to match *exactly* to
# see each other's messages, so this needs to be the plain "ME193"
# everyone else is already on, not a more specific topic like
# "ME193/chat" (tried that -- it just meant nobody else's messages
# (or mqtt_test.py's own plain-text ones) ever arrived). That also
# means this is the same public, crowded, unauthenticated topic
# everyone in class (and test.mosquitto.org at large) is using -- see
# MQTTLIB.md's security notes. Plain-text messages from non-chat
# clients (e.g. mqtt_test.py) still show up fine, just as an anonymous
# "?" sender, via the fallback in _handle_message below.
TOPIC = "ME193"

BG_COLOR = "#ffffff"
MY_BUBBLE_COLOR = "#0b93f6"
MY_TEXT_COLOR = "white"
OTHER_BUBBLE_COLOR = "#e5e5ea"
OTHER_TEXT_COLOR = "black"


class ChatApp:
    def __init__(self, root, client, username):
        self.root = root
        self.client = client
        self.username = username

        root.title(f"ME193 MQTT Chat -- {username}")
        root.geometry("420x560")
        root.configure(bg=BG_COLOR)

        tk.Label(root, text=f"Topic: {TOPIC}  (public broker -- see MQTTLIB.md)",
                 bg=BG_COLOR, fg="#888888", font=("Segoe UI", 8)).pack(side="top", pady=(6, 0))

        # Scrollable message area: a Canvas holding a Frame that grows
        # downward as bubbles are added, with a Scrollbar wired to it.
        body = tk.Frame(root, bg=BG_COLOR)
        body.pack(side="top", fill="both", expand=True, padx=(8, 0), pady=8)

        self.canvas = tk.Canvas(body, bg=BG_COLOR, highlightthickness=0)
        scrollbar = tk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.messages_frame = tk.Frame(self.canvas, bg=BG_COLOR)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.messages_frame, anchor="nw")
        self.messages_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind(
            "<Configure>", lambda e: self.canvas.itemconfigure(self.canvas_window, width=e.width))

        # Entry row
        entry_row = tk.Frame(root, bg=BG_COLOR)
        entry_row.pack(side="bottom", fill="x", padx=8, pady=8)

        self.entry = tk.Entry(entry_row, font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", self._on_send)
        self.entry.focus_set()

        tk.Button(entry_row, text="Send", command=self._on_send).pack(side="left", padx=(6, 0))

        # mqttlib's subscribe() callback fires on its own background
        # network thread, not Tkinter's main thread -- see
        # _on_mqtt_message below for why that matters.
        self.client.subscribe(TOPIC, self._on_mqtt_message)

    def _on_send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.client.publish(TOPIC, json.dumps({"sender": self.username, "text": text}))

    def _on_mqtt_message(self, topic, payload):
        # Tkinter widgets can only be touched from the main/UI thread,
        # but this runs on mqttlib's background network thread --
        # root.after() hands the actual work back to the UI thread
        # instead of touching widgets directly from here.
        self.root.after(0, self._handle_message, payload)

    def _handle_message(self, payload):
        try:
            data = json.loads(payload)
            sender, text = data["sender"], data["text"]
        except (ValueError, KeyError, TypeError):
            sender, text = "?", payload  # tolerate a plain-text sender, e.g. mqtt_test.py
        self._add_bubble(sender, text, is_me=(sender == self.username))

    def _add_bubble(self, sender, text, is_me):
        row = tk.Frame(self.messages_frame, bg=BG_COLOR)
        row.pack(fill="x", padx=4, pady=4)

        bubble_color = MY_BUBBLE_COLOR if is_me else OTHER_BUBBLE_COLOR
        text_color = MY_TEXT_COLOR if is_me else OTHER_TEXT_COLOR

        bubble = tk.Frame(row, bg=bubble_color)
        bubble.pack(anchor="e" if is_me else "w")

        if not is_me:
            tk.Label(bubble, text=sender, bg=bubble_color, fg="#555555",
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(4, 0))

        tk.Label(bubble, text=text, bg=bubble_color, fg=text_color, wraplength=260,
                 justify="left", font=("Segoe UI", 11)).pack(anchor="w", padx=10, pady=(0, 6) if not is_me else 6)

        self.root.after_idle(lambda: self.canvas.yview_moveto(1.0))  # auto-scroll to newest


def main():
    root = tk.Tk()
    root.withdraw()  # hide the main window until a name is chosen
    username = simpledialog.askstring("ME193 MQTT Chat", "Your name:", parent=root)
    if not username:
        return
    root.deiconify()

    with MQTTClient() as client:
        ChatApp(root, client, username)
        root.mainloop()


if __name__ == "__main__":
    main()
