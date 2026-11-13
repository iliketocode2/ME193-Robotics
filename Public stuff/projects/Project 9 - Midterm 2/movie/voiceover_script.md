# Voice-over script for the stop-action movie

The video `nn_movie.mp4` is 3:33.1 long, silent, at 8 frames a second. Each scene below gives its start and end time and the words to read over it. Every scene is long enough to read its part at about 150 words a minute, with a short pause at the end.

## Recording tips

- Play the video full screen on one monitor and record yourself on Zoom (or any recorder), starting the recording at the same moment you press play.
- Read each part when its timestamp comes up. If you finish early, stop and wait for the next scene, so the voice stays lined up with the pictures.
- To put the audio on the video, drop both into Clipchamp or iMovie, or run this from the `movie` folder (it needs ffmpeg; the copy inside the Python environment works too):

```
ffmpeg -i nn_movie.mp4 -i voiceover.m4a -map 0:v -map 1:a -c:v copy -c:a aac -shortest nn_movie_voiced.mp4
```

- A quiet room and a headset mic sound better than a laptop mic.

## Script

### 0:00.0 to 0:12.0: Title

This is my finger-count lock. A webcam looks down at my hand, and a small neural network counts my fingers, from zero to five. Here's how it works.

### 0:12.0 to 0:32.8: From camera to numbers

The program first keeps the center square, turns it gray and shrinks it to 64 by 64 pixels, so it's just a grid of numbers from 0 to 255. Then it subtracts the average brightness and divides by the spread, so a bright room and a dim room look alike.

### 0:32.8 to 1:03.5: Convolution

The first layer is a convolution. A filter is a 5 by 5 grid of weights the network learned. It slides over the image, and at each spot it multiplies the 25 pixels by the 25 weights, adds them up and adds a bias. A patch that matches the pattern gives a big positive number, in red, and the opposite pattern gives a negative one, in blue. All the spots together make a feature map.

### 1:03.5 to 1:15.5: Eight filters and ReLU

There are eight filters, so one photo becomes eight feature maps, each picking out something different. Then ReLU keeps the positive numbers and turns every negative into zero.

### 1:15.5 to 1:33.1: Max pooling

Next comes max pooling. A 2 by 2 window steps across the map and keeps the biggest of each four numbers. That halves the size to 32 by 32, keeps the strong responses, and makes small shifts of my hand matter less.

### 1:33.1 to 1:49.9: Going deeper

Convolution, ReLU and pooling happen three times. The second layer's 16 filters each look at all eight maps before them, so they can combine edges into pieces of fingers. The third layer's 32 filters tend to respond to whole fingers.

### 1:49.9 to 2:10.8: From maps to an answer

Then the 32 small maps become one line of 2048 numbers. Each count has its own 2048 weights, and multiplying and adding gives six scores. Softmax turns them into probabilities that add up to one. The tallest bar is the answer, and its height is how sure the network is.

### 2:10.8 to 2:28.8: Where it looked

Since the last layer comes right after the third convolution, I can split each score back onto the 8 by 8 grid. Red shows what pushed the network toward its answer. Often that's the fingers, and sometimes it's the empty space beside them.

### 2:28.8 to 3:04.8: How it learned

The weights start out random, so at first the network just guesses. The loss is minus the log of the probability it gave the right answer. Backpropagation uses the chain rule, like my V-fit spreadsheet, to find how each of the 18,310 weights changes the loss, and gradient descent nudges them all downhill. Each training photo is also randomly rotated, shifted, zoomed, flipped and brightened. In the end it got 98 percent of the test photos right, though they came from the same session as the training photos.

### 3:04.8 to 3:29.1: From counts to the lock

Finally, the lock. I press the button and take my hand away so it reads zero. Then I hold up a count for a second at 80 percent or more, and a yellow LED lights. After three digits the whole code is checked at once, and the right code turns on the green LED and swings the latch open.

### 3:29.1 to 3:33.1: End card

(no narration, let the end card play)
