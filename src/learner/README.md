# This is the ML pipeline for detecting the ping pong ball.

## image_extractor

Takes a `.bag` file as input, and generates pairs of depth/ir images.

## labeller

Takes a directory as input, shows window and annotates the position of the ping pong ball.

The binary creates ts_label.txt, each line contains comma separated x, y and r.

## trainer

As for representing the position of the ball, we split the image to 21x12 of 40x40 grids, and for each grid, we have a vector of 4 numbers: [prob, x, y, r]. x,y are relative coordinates in the sub grid, while r is the radius.