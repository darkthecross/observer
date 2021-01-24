Experimental code for capturing motion with a realsense camera. 

## Dependencies

Boost, Opencv, librealsense, opengl/glew/glfw, gflags, glogs.

### install

This bad boy installs all other dependencies besides librealsense and cuda. 

```
sudo sh install.sh
```

This one installs realsense on Nvidia Jetson Xavier:

```
install_jetson.sh
```

These scripts downloads >2G data from internet (500G for opencv, 1.4G for cuda) and runs >1 hour on Nvidia Jetson Xavier (compiling opencv from source takes some time.)

TODO(darkthecross): add script for installing realsense and cuda on ubuntu.

## Reference

### High speed mode for realsense

Please refer to this [doc](https://dev.intelrealsense.com/docs/high-speed-capture-mode-of-intel-realsense-depth-camera-d435).

