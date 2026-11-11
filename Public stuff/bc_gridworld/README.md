# Clone the Navigator (grid-world behavior cloning)

A browser version of the unplugged "navigate the crowded room" activity. No install, and it works in any browser.

**Open it:** [chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/bc_gridworld/](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/bc_gridworld/), or open `index.html` from a local copy.

**Printables** for the unplugged version: [activity guide (PDF)](print/clone-the-navigator-guide.pdf) and [scribe worksheet (PDF)](print/clone-the-navigator-scribe.pdf). Edit `print/guide.html` or `print/scribe.html` and print to PDF from Chrome to change them.

- **Tab 1, Demonstrate:** you are the expert. Arrow keys (or W/A/D) drive; every key press records a card (what the robot sensed, what you did).
- **Tab 2, Run the clone:** press **Train** to build the lookup table from the cards, then **Run** or **Step**. Press a key during a run to take over; that records an orange correction card (DAgger). Train again and re-run.
- The robot senses only Front/Left/Right blocked (optionally also the goal direction). Cards persist in the browser and carry across rooms, so you can train on one room and test on another.
- "Add scripted expert cards (instructor)" adds cards from a scripted expert that sees the whole map, so you can demo without driving.

Files: `logic.js` (rules, no page code), `index.html` (the page), `tests/` (serve this folder with `python3 -m http.server`, then open `/tests/tests.html` and `/tests/page_test.html` in Chrome; each prints ALL PASSED), `print/` (the activity guide and scribe worksheet as HTML and PDF).
