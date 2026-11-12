# Reward lab

### 👉 [Click here to try it](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/reward-lab/)

`index.html` is a single-page reward-design exercise: a robot on a grid maze
always takes the move with the best *immediate* reward. It starts with the
"straight-line distance to the goal" reward. Predict where it gets stuck, run
it, then change the reward and see what happens. Nothing to install: it runs
entirely in the browser, so you can open it online (above) or double-click the
file.

## How to use it

- **Edit the map** — toggle walls, and move the start and goal.
- **Step or Run** the robot, and watch what it is weighing at each move.
- **Presets or your own reward** — pick a preset (such as shortest-path
  distance), or build one with the sliders (revisit penalty, time penalty,
  goal bonus, and so on). The page shows your reward as Python code.
- **Break ties randomly** — otherwise the robot takes the first best move in
  the order right, left, up, down.

The page ends with a "Things to try" list of questions to work through.
