# Project 7 – Activity Wearable

An Arduino UNO Q with an MPU-6050 IMU, strapped to the top of a shoe. It tells
apart **standing still**, **walking**, **running** and **biking** (or lying on
your back and air-biking) with two models. Neither one is a neural net:

| | Input | Classifier |
|---|---|---|
| **Model 1** | Peak acceleration along **one axis** in each 2 s window (largest \|a − mean\|, so gravity drops out) | One Gaussian per activity over that single number; the most likely activity wins. The axis is whichever separates the activities best in your training data (shown on the dashboard). |
| **Model 2** | **All IMU outputs**: ax, ay, az, gx, gy, gz, plus \|a\| and \|g\|. Each window is summarised as mean / std / min / max per channel, plus step cadence (the dominant frequency of \|a\|). That is 34 features. | k-nearest-neighbours (k = 5) on standardised features. |

Both models classify a sliding 2 s window every 0.5 s. The label shown is the
most common of the last three predictions, so it doesn't flicker.

## Files

```
unoq_app/                    the App Lab app (copy into your ArduinoApps folder)
├── app.yaml                 uses the web_ui brick (page on port 7000)
├── sketch/sketch.ino        MCU: reads the MPU-6050 at 100 Hz, sends batches over the Bridge
├── python/main.py           Linux side: recording, training, live prediction, talks to the page
├── python/activity.py       features + both models (pure numpy, testable on a laptop)
└── assets/                  the dashboard (index.html, app.js, style.css)
test_activity_offline.py     laptop check of activity.py on fake data
```

## Wiring

Connect the MPU-6050 to the UNO Q's **Qwiic** connector, or to the SDA/SCL/3V3/GND
header pins. The sketch searches both buses at addresses 0x68 and 0x69, so either
works. Mount the board firmly on top of the shoe; how you orient it doesn't matter
as long as it stays the same between training and testing. Power it from a USB-C
battery pack.

## Running it

1. Copy `unoq_app/` into your `ArduinoApps` repo (for example as `activity-wearable/`), open any file in it, and press **Ctrl+Shift+B → Run on UNO Q**. You can also run `python tools/deploy.py activity-wearable`.
2. Open `http://<board IP>:7000` on your laptop or phone (same network). The **IMU** pill should go green and the charts should move.
3. **Collect training data.** Tap an activity, do it for 20–60 s, then tap **Stop**. The first and last 2 s are dropped automatically. Record each activity two or three times. Use **delete** on an activity to throw out its recordings and redo them.
4. Tap **Train both models**. The page shows each model's accuracy and confusion matrix on held-out data (the last 25 % of every recording, which isn't used for fitting). The models are then refit on everything and saved to `models.json`, so they survive a restart.
5. Move around. Both predictions update every 0.5 s. Model 1's number line shows where each activity's typical peak sits and where your current peak is.
6. Open the **Feature map** tab (`map.html`). Every training window is a faint dot and you are the big moving dot.
   - The x-axis is Model 1's only input.
   - The y-axis is the Model 2 feature that best separates the two activities Model 1 confuses most.
   - The strip underneath shows the same points on the x-axis alone, which is all Model 1 can see.
   - Clusters that overlap left to right but separate top to bottom are where Model 2 beats Model 1.
   - The dropdowns let you plot any pair of features.

Recordings are saved as CSV in `data/` inside the app folder on the board. Deploying
never deletes them.

## Offline check (no board)

```
my_env\Scripts\python "Public stuff/projects/Project 7 - Wearable/test_activity_offline.py"
```

This trains both models on synthetic gait data and prints their accuracy. It only
proves the pipeline runs; real data is messier.
