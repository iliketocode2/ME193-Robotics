# Project 9: Midterm 2, the finger-count lock

A lock that opens when you show the camera the right sequence of three finger counts. A convolutional neural
network, written in PyTorch and trained on photos of my own hand, reads how many fingers are showing. It runs on
the computer first, with a USB webcam, and then on the Arduino UNO Q.

## Contents

1. [Part 1: fitting the V data](#part-1-fitting-the-v-data)
2. [Part 2: the finger-counting network](#part-2-the-finger-counting-network)
3. [Part 3: the lock policy](#part-3-the-lock-policy)
4. [Running it](#running-it)
5. [Collecting good data](#collecting-good-data)
6. [Files](#files)

## Part 1: fitting the V data

`v_fit/` holds the gradient-descent warm-up: the V data from Anna's Code (`make_v_data.py`, `v_data.tsv`) and
`V Fit Gradient Descent.xlsx`, which trains a straight line, one ReLU neuron and a layer of two ReLU neurons with
spreadsheet formulas. The line can only flatten out through the middle of the V, one ReLU neuron fits one arm,
and two ReLU neurons fit the whole V.

## Part 2: the finger-counting network

The camera looks down at the spot where you hold your hand. The program cuts out the center square of each frame,
turns it gray, and shrinks it to 64 x 64 pixels. Each image is then standardized (its mean brightness subtracted
and divided by its spread) so a brighter or darker room changes the numbers less.

FingerNet (`model.py`) has three convolution layers, each followed by ReLU and 2 x 2 max pooling, and one linear
layer at the end:

| Layer | What it does | Output size |
| --- | --- | --- |
| input | grayscale image | 1 x 64 x 64 |
| conv1 | 8 filters, 5 x 5, then ReLU and pooling | 8 x 32 x 32 |
| conv2 | 16 filters, 3 x 3, then ReLU and pooling | 16 x 16 x 16 |
| conv3 | 32 filters, 3 x 3, then ReLU and pooling | 32 x 8 x 8 |
| linear | 2048 numbers in, one score per answer | 6 |

A convolution filter is a small grid of weights that slides across the image and gives a large output wherever
the patch under it matches its pattern. The first layer learns simple patterns such as edges between skin and
background. Later layers combine those into finger-shaped pieces. Max pooling keeps the strongest response in each
2 x 2 block, which halves the image size and lets later layers see a wider area. The final linear layer weighs all
2048 values and gives a score to each answer from 0 to 5 fingers. Softmax turns the scores into probabilities.

The network has about 18,000 weights. Training uses cross-entropy loss, which is small when the network puts a high
probability on the right count, because this is a choice between six categories rather than a number to estimate.
`loss.backward()` runs backpropagation, the chain rule from the V-fit spreadsheet applied through every layer,
and the Adam optimizer takes the gradient-descent step.

Augmentation (`dataset.augment`) makes every training batch a little different. Each photo is rotated up to 20
degrees, shifted up to 12% of the frame, zoomed between 85% and 115%, mirrored half the time (a left hand becomes
a right hand), and given a random brightness and contrast change. None of these change the number of fingers,
so each altered copy is a new, correctly labeled example. `python train.py --no-augment` trains the same network
without it, to measure how much it helps.

For the UNO Q, `npnet.py` repeats the same forward pass in plain numpy (convolution, ReLU, pooling, linear,
softmax), so the board does not need PyTorch. `train.py` checks that both versions give the same probabilities.

## Part 3: the lock policy

`lock_policy.py` holds the rules and nothing else, so the same file runs on the computer and on the UNO Q. It gets
one reading per camera frame (the most likely count and how sure the network is) plus button presses, and it says
which LEDs to light and when the latch opens. The settings are in `config.py`.

An attempt starts with the button. Before each digit the hand has to leave the view, so the network reads 0 for
0.4 seconds. Then you show a count from 1 to 5 and hold it. The digit is accepted when the network gives the same
count, at least 85% sure, for one full second without a break, and one digit LED lights. A single wrong frame
restarts the second. Requiring the empty view between digits is what lets a code repeat a digit, such as 3, 3, 5,
and it stops one long hold from being read as two digits.

After the third digit the whole code is checked at once, so a wrong attempt gives no hint about which digit was
wrong. The right code opens the latch for 5 seconds and then it locks itself. A wrong code shows red for a moment.
Three wrong codes in a row start a 30 second lockout during which the button does nothing. Ten seconds without a
new digit cancels the attempt, and pressing the button mid-attempt cancels it too.

`lock.py` runs the policy with the webcam. SPACE stands in for the button, three white dots stand in for the digit
LEDs and a large dot is the status LED. Each event goes into `results/lock_log.csv` with its time, which gives
the time taken to unlock. The digits are never logged.

## Running it

```bash
pip install -r requirements.txt

python collect.py --list        # find the webcam's index, then set CAMERA_INDEX in config.py
python collect.py               # photograph your hand; 0-5 picks the label, SPACE takes a burst
python train.py                 # about 40 epochs; writes models/, plots/ and results/
python train.py --no-augment    # optional: the same network without augmentation, for comparison
python live.py                  # live finger count; press f to see the feature maps
python live.py --numpy          # the same, using the numpy network the UNO Q will run
python lock.py                  # the lock; SPACE is the button
python lock.py --hide           # the lock without the live reading on screen
```

`train.py` saves these plots:

| Plot | Shows |
| --- | --- |
| `plots/training_curves.png` | loss and accuracy for each epoch, training and validation |
| `plots/confusion.png` | which counts get mixed up with which |
| `plots/mistakes.png` | the validation photos it got wrong |
| `plots/augmentation_examples.png` | one photo per count and five random altered copies |
| `plots/conv1_filters.png` | the eight 5 x 5 filters the first layer learned |
| `plots/feature_maps.png` | one photo passing through each layer, and the final probabilities |

## Collecting good data

- Use the USB webcam, mounted where it will sit on the safe, looking down at a plain background.
- Take at least two sessions for every label, on different days or in different light. `train.py` holds out each
  label's newest session for validation. Photos from one burst are near copies of each other, so testing on the
  same session would overstate the accuracy.
- Aim for 150 to 300 photos per label. During a burst, move your hand: closer and farther, tilted, left hand and
  right hand, different finger combinations for the same count (for 2, both index and middle, and thumb and pinky).
- Label 0 covers a fist and the empty view with no hand, so the lock can tell when you take your hand away.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | camera index, crop size, image size, paths |
| `camera.py` | opens the camera and turns a frame into the network input (OpenCV and numpy only) |
| `collect.py` | labeled photo capture |
| `dataset.py` | loading, the session-based split, and augmentation |
| `model.py` | FingerNet in PyTorch, checkpoint saving, numpy weight export |
| `npnet.py` | the same network's forward pass in numpy, for the UNO Q |
| `train.py` | training, plots, and the numpy check |
| `live.py` | live finger count with probabilities and feature maps |
| `lock_policy.py` | the lock's rules: digit acceptance, code check, timeout, lockout |
| `lock.py` | the lock on the computer, with on-screen LEDs and an event log |
| `v_fit/` | Part 1, the V-data gradient-descent spreadsheet |
