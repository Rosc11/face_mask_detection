#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import pyds
import cv2
import numpy as np
from nms import nms

SHOW_STREAM = True

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        detected_rects = []
        scores = []
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)

            #add detected rect
            score = obj_meta.confidence
            c = obj_meta.class_id
            x = obj_meta.rect_params.left
            y = obj_meta.rect_params.top
            w = obj_meta.rect_params.width
            h = obj_meta.rect_params.height
            detected_rects.append([c,x,y,w,h])
            scores.append(score)

            try:
                l_obj=l_obj.next
            except StopIteration:
                break

        #apply nms
        det_rects = np.array(detected_rects)
        scores = np.array(scores)
        rects = [rect[1:5] for rect in det_rects]
        indices = nms.boxes(rects, scores)
        detected_rects = det_rects[indices]
        print("Frame " + str(frame_number)  + ", " + str(len(detected_rects))+" object(s) detected!")

        #retrieve & save image
        if (len(detected_rects) > 0):
            # the input should be address of buffer and batch_id
            n_frame=pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
            #convert python array into numy array format.
            frame_image=np.array(n_frame,copy=True,order='C')
            #covert the array into cv2 default color format
            frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)
            for i in range(0, len(detected_rects)):
                c = detected_rects[i][0]
                rect = detected_rects[i][1:5].astype(np.int64)
                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]
                frame_crop=frame_image[y:y+h, x:x+w]
                
                #save image
                label = "mask" if c==0 else "no-mask"
                #cv2.imwrite("imgs/frame_"+str(frame_number)+"_crop_"+str(i)+"_"+label+".jpg",frame_crop)
        try:
            l_frame=l_frame.next
        except StopIteration:
            print("ERROR at collecting frame")
            break
    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) == 1:
        SHOW_STREAM = True
    if len(args) == 2:
        if args[1] == "no-stream":
            SHOW_STREAM = False
    elif len(args) > 2:
        sys.stderr.write("usage: " + args[0] + " \n no-stream [optional]")
        sys.exit(1)


    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)
    
    # We need to create a pipeline to get the images from the raspberry pi v2 camera
    # Convert command to code:
    # command = gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! ximagesink -e
    # VERY IMPORTANT: since deepstream is used, we need to use nv methods instead of regular methods

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading the Raspberry pi camera v2
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("nvarguscamerasrc", "cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")
    
    # We need to tell which input we want (resolution, framerate, format) with a capsfilter
    # " video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12" "
    caps_input = Gst.ElementFactory.make("capsfilter", "input_caps")
    if not caps_input:
        sys.stderr.write(" Unable to create capsfilter \n")

    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    # A capsfilter element to convert image resolution to network resolution
    caps_convertres = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_convertres:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # 
    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    # This is where the probe is placed to draw on images
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    # Finally render the osd output
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    if (SHOW_STREAM):

        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")
    else:
        sink = Gst.ElementFactory.make("fakesink", "fakesink")

    #caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=21/1, format=NV12"))
    caps_input.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12"))
    caps_convertres.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=960, height=544"))
    source.set_property('bufapi-version', 1)
    streammux.set_property('width', 960)
    streammux.set_property('height', 544)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "mask_detector_config.txt")
    if (SHOW_STREAM):
       # Set sync = false to avoid late frame drops at the display-sink
       sink.set_property('sync', False)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_input)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_convertres)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)

    # we link the elements together
    # source -> converters -> mux ->
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")

    source.link(caps_input)
    caps_input.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_convertres)
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_convertres.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64() and SHOW_STREAM:
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
        print("loop started!")
    except :
        pass
    # cleanup
    print("Exiting app")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

