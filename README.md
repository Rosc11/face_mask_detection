# Face mask detection on a NVIDIA Jetson nano using DeepStream SDK

Prequisites:

* DeepStreamSDK 5.0
* Jetpack 4.4.1 (L4T 32.4.4)
* Python 3.6
* libjpeg-dev (apt-get install)

Deepstream install: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#jetson-setup

pip packages:

* cython
* numpy
* nms
* cv2
* Pillow
* aiohttp

To run the app:
```
    python3 mask_detector_cam.py \
       no-stream [optional]                 (disables screen display of the stream) 
```
Make sure all the paths are configured as they should be (if it doesn't run, check camera.py on line 204 - pgie.set_property, the config file and start.sh)

Or you can run the script with:

```
./start.sh
```
This shell command will automatically change your working directory to face_mask_detection (used for systemd service, as mentioned later in this readme).

## Test the raspberry pi v2 camera
To test raspberry pi v2 camera only, run cam_test.py in /tests dir
```
    python3 cam_test.py
```
## How it works
This repository makes use of NVIDIA Deepstream SDK for inferring a trained Detectnet-v2 network and gstreamer to capture images from a raspberry pi v2 camera.

The Gst pipeline is built with the following components:

*  nvaruguscamerasrc: 
    *  The source for the raspberry pi v2 camera.
*  capsfilter: 
    *  Needed to specify which input we want to select from the raspberry pi v2 camera.
    *  Options (print of nvarguscamerasrc):
```
GST_ARGUS: Available Sensor modes :
GST_ARGUS: 3264 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 3264 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 1640 x 1232 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 1280 x 720 FR = 120.000005 fps Duration = 8333333 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
```
*  videoconvert followed by nvvideoconvert
    *  Not all raw formats are supported by nvvideoconvert -> videoconvert is needed.
    *  nvvideoconvert converts incoming raw buffers to NVMM Mem (NvBufSurface API).
*  capsfilter (caps_convertres)
    *  A filter to convert images to network resolution for inference.
*  nvstreammux
    *  Batch forming of images.
*  nvinfer
    *  The detection network.
*  nvvidconv
    *  Converts from NV12 format to RGBA (so nvosd & opencv can read the images).
*  nvosd
    *  Creates an OSD to draw on the converted RGBA buffers.
    *  At this, we place a probe to gather detection information.
*  nvegltransform
    *  Needed to render the osd output in an output window
*  nveglglessink / fakesink
    *  nveglglessink = the output window
    *  fakesink = a pipeline always needs to end with a sink -> a fake sink enables to only process the images without displaying them

### Using detections
To gather detection information, a probe is placed on nvosd. This probe does not interfere with the information processed in the pipeline. It only gains access to frame meta information such as bounding boxes & the frame itself. This means applying nms does not filter the bounding boxes in the pipeline, which is why all predicted bounding boxes are still shown in nveglessink. Nevertheless, bounding boxes and frame objects are loaded in parameters for custom processing.

### Models
Models were created using https://github.com/NVIDIA-AI-IOT/face-mask-detection . When creating TensorRT engine files for Jetson Nano, make sure you're using the right version of tlt-converter for your platform (depends on your TensorRT version). This repository made use of TensorRT 7.1 (downloads at https://developer.nvidia.com/tlt-getting-started).
  
### Doing your own stuff with this repository
You can work further with detections & scores in osd_sink_pad_buffer_probe in camera.py (after nms is applied at line 75!)

## Run script in LXDE-GUI mode
For maximum performance while viewing the camera output, we recommend to start the jetson nano GUI in LXDE mode (https://www.jetsonhacks.com/2020/11/07/save-1gb-of-memory-use-lxde-on-your-jetson/)
When using a jetpack version > 4.4, LXDE should be already installed in the jetson nano. You can enable it in gdm3 mode at login. When selected, switch to lightdm mode.

If LXDE is not installed: install LXDE with:
```
git clone https://github.com/jetsonhacks/installLXDE.git
cd installLXDE
./installLXDE.sh
```

Then, make sure to boot into lightdm mode by selecting lightdm after the following command:
```
sudo dpkg-reconfigure lightdm
```
Reboot your Jetson Nano for the changes to take effect

### Run script at boot using a systemd service

** IMPORTANT: When trying to execute script at boot, make sure ALL the paths in the script and config file are absolute paths!**

Create a new systemd service by doing:

```
sudo nano /etc/systemd/system/facemasks.service
```

Then, paste the following in the editor:
```
[Unit]
Description=facemask detection
After=graphical.target

[Service]
User=<user>
Group=<group>
ExecStart=<path to start.sh>
Restart=always
StartLimitInterval=10
RestartSec=10

[Install]
WantedBy=graphical.target
```

Save the file. Enable and start the service by executing:
```
sudo systemctl enable facemasks.service
sudo systemctl start facemasks.service
```

Now the script will start automatically at boot!


### Important notes
* This pipeline is built up with the nvarguscamerasrc command in mind:
 ```
 gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! ximagesink -e
 ```
 
* As for now, either one of "videoflip" with "method=rotate-180" or "nvvidconv" with "flip-method=2" doesn't work when building a pipeline with nvinfer. Reason unknown at this moment.
* When working with Python Gst in combination with Deepstream, nv- elements are best use (CUDA accelerated gstreamer elements). Consistency is key
    * though, the very first videoconvert is needed to support raw formats
    
## Extra documenation
For more info, visit the NVIDIA accelerated GStreamer user guide at https://developer.download.nvidia.com/embedded/L4T/r32_Release_v1.0/Docs/Accelerated_GStreamer_User_Guide.pdf

