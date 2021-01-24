# Realsense
mkdir install_jetson && cd install_jetson
git clone https://github.com/jetsonhacks/installRealSenseSDK.git
cd installRealSenseSDK
sudo sh installLibrealsense.sh

cd ..
sudo rm -R install_jetson/