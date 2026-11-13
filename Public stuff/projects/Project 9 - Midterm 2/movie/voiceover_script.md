# Voice-over script for the stop-action movie

The video `nn_movie.mp4` is 2:29.9 long, silent, at 8 frames a second. Each scene below gives its start and end time and the words to read over it. Every scene is long enough to read its part at about 150 words a minute, with a short pause at the end.

## Recording tips

- Play the video full screen on one monitor and record yourself on Zoom (or any recorder), starting the recording at the same moment you press play.
- Read each part when its timestamp comes up. If you finish early, stop and wait for the next scene, so the voice stays lined up with the pictures.
- To put the audio on the video, drop both into Clipchamp or iMovie, or run this from the `movie` folder (it needs ffmpeg; the copy inside the Python environment works too):

```
ffmpeg -i nn_movie.mp4 -i voiceover.m4a -map 0:v -map 1:a -c:v copy -c:a aac -shortest nn_movie_voiced.mp4
```

- A quiet room and a headset mic sound better than a laptop mic.

## Script

### 0:00.0 to 0:06.2: Title

A small neural network in my lock counts fingers from zero to five. Here's how it works.

### 0:06.2 to 0:20.2: From camera to numbers

First, the program keeps the center square, turns it gray and shrinks it to 64 by 64, so it's just numbers from 0 to 255. Then it subtracts the average brightness and divides by the spread.

### 0:20.2 to 0:42.8: Convolution

Convolution comes first. A filter is a 5 by 5 grid of learned weights. It slides over the image, multiplies the 25 pixels by the 25 weights and adds them up, plus a bias. A matching patch gives a big positive number, in green, and the opposite gives a negative one, in red. All of them make a feature map.

### 0:42.8 to 0:51.2: Eight filters and ReLU

Eight filters turn one photo into eight feature maps. Then ReLU keeps the positive numbers and turns every negative into zero.

### 0:51.2 to 1:04.5: Max pooling

Max pooling slides a 2 by 2 window across each map and keeps the biggest number. That halves the size to 32 by 32 and makes small shifts of my hand matter less.

### 1:04.5 to 1:17.4: Going deeper

This repeats three times. A layer's width is its number of filters: 8, then 16, then 32. With the output layer the network is four layers deep, and deeper filters combine edges into fingers.

### 1:17.4 to 1:34.0: From maps to an answer

The 32 small maps become one line of 2048 numbers. Each count has 2048 weights, and multiplying and adding gives six scores. Softmax turns them into probabilities. The tallest bar is the answer, and its height is how sure it is.

### 1:34.0 to 1:45.5: Where it looked

Because that last step is just weights times values, I can split each score back onto an 8 by 8 grid. Green shows what pushed the network toward its answer.

### 1:45.5 to 2:09.8: How it learned

The weights start random. The loss is minus the log of the probability given to the right answer. Backpropagation uses the chain rule, like my V-fit spreadsheet, to find each weight's slope, and gradient descent steps all 18,310 weights downhill. Rotated, shifted and flipped copies of each photo add more examples. In the end it got 98 percent of the test photos right.

### 2:09.8 to 2:26.1: From counts to the lock

Finally, the lock. I press the button and show an empty view, so it reads zero. Then I hold a count for one second at 80 percent or more, and a yellow LED lights. After three digits, the right code opens the latch.

### 2:26.1 to 2:29.9: End card

(no narration, let the end card play)
