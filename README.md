# Face mask detection on a NVIDIA Jetson nano using DeepStream SDK

Prequisites:
* DeepStreamSDK 5.0
* Python 3.6
* Gst-python
* nms
* numpy
* cv2

To run the app:
```
    python3 mask_detector_cam.py \
       no-stream [optional]                 (disables screen display of the stream) 
```

To test raspberry pi v2 camera only, run cam_test.py in /tests folder
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
  
    
## Doing your own stuff with this repository
You can work further with detections & scores in osd_sink_pad_buffer_probe in camera.py (after nms is applied at line 75!)

## Important notes
* This pipeline is built up with the nvarguscamerasrc command in mind:
 ```
 gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! ximagesink -e
 ```
* As for now, either one of "videoflip" with "method=rotate-180" or "nvvidconv" with "flip-method=2" doesn't work when building a pipeline with nvinfer. Reason unknown at this moment.
* When working with Python Gst in combination with Deepstream, nv- elements are best use (CUDA accelerated gstreamer elements). Consistency is key
    * though, the very first videoconvert is needed to support raw formats
    
## Extra documenation
For more info, visit the NVIDIA accelerated GStreamer user guide at https://developer.download.nvidia.com/embedded/L4T/r32_Release_v1.0/Docs/Accelerated_GStreamer_User_Guide.pdf