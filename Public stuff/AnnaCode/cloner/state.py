"""Shared state for the Table Cloner page.

Device.py and power.py (shared with the main page) read a few names from here, so this stands in
for its much bigger state.py. Other modules use `state.x`, never
`from state import x`, so changes are visible everywhere.
"""
from pyscript import document

devices: list = []

# misc_ui/power.py (Save power) is shared with the main page and reads these names.
# The Cloner's three sensor dropdowns play the part of the main page's Input nodes
# (their element ids are dev-in-left / -front / -right), and Record / Run play the part of
# Record / Run.
inputs: list = [{"id": "left"}, {"id": "front"}, {"id": "right"}]
outputs: list = []
is_running = False           # Run table is on
is_recording = False         # Record is on


def get_id(id_: str):
    return document.getElementById(id_)
