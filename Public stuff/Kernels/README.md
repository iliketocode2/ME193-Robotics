# Kernels

### 👉 [Click here to try it](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/Kernels/)

`index.html` is a single-page playground for exploring image convolution.
Nothing to install, no server: it runs entirely in the browser, so you can
open it online (above) or double-click the file.

## How to use it

1. **Image** — drop in an image, choose a file, or paste one from the
   clipboard (Ctrl+V / Cmd+V). Large images are scaled down to keep things fast.
2. **Threshold** — optionally turn the image into black and white: pixels
   brighter than the cutoff go white, the rest go black.
3. **Kernel** — pick a preset (box blur, Gaussian blur, sharpen, edge detect,
   Sobel X/Y, emboss) or type your own 3×3 values, then press **Convolve**.
   "Normalize by kernel sum" keeps the overall brightness the same.

The original, thresholded and convolved images are shown side by side, with a
large view for a closer look.
