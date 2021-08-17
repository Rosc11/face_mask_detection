import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

from utils import save_face_crops, get_cv2_image, apply_nms, resize_image_post_request
from api import send
from api_async import send_async
from concurrent.futures import ProcessPoolExecutor
import pyds
import time
import asyncio
import parameters
import threading
import queue
import cv2
from PIL import Image
import aiohttp
import json
import base64


ENDPOINT = parameters.ENDPOINT

class Camera:
    def __init__(self, show_stream):
        self.pipeline = None
        self.loop = None
        self.notsend = False
        self.last_time_sent = 0
        self.send_interval = parameters.SEND_INTERVAL
        self.started = False
        self.queue = queue.Queue()
        self.show_stream = show_stream

    def osd_sink_pad_buffer_probe(self, pad,info,u_data):
        if (not self.started):
            self.started = True
            print('Buffer now running!')
#        print("Entered buffer probe.")
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
            #scores = []
            i=0
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                    obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                #obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)

                #add detected rect
                score = obj_meta.confidence
                c = obj_meta.class_id
                x = obj_meta.rect_params.left
                y = obj_meta.rect_params.top
                w = obj_meta.rect_params.width
                h = obj_meta.rect_params.height
                detected_rects.append([c,score,x,y,w,h])
                #scores.append(score)
                i = i + 1
                try:
                    l_obj=l_obj.next
                except StopIteration:
                    break
            #apply nms
            detected_rects = apply_nms(detected_rects)
#            print("Frame " + str(frame_number)  + ", " + str(len(detected_rects))+" object(s) detected!")

            #retrieve & save image
            if (len(detected_rects) > 0):
                now = round(time.time() * 1000)
                diff = now - self.last_time_sent
                if diff > self.send_interval:
                    self.last_time_sent = now
                    frame_image = get_cv2_image(gst_buffer, frame_meta)
#                    frame_image = resize_image_post_request(frame_image, 640, 480)
                    print("now sending frame " + str(frame_number) + " with timestamp " + str(now))
                    before = round(time.time() * 1000)
                    send(now, frame_image, detected_rects)
                    now = round(time.time() * 1000)
                    diff = now - before
                    print("API call took " + str(diff) + " milliseconds")
#                    self.queue.put((now, frame_image, detected_rects))
#                    loop = asyncio.new_event_loop()
#                    asyncio.set_event_loop(loop)
#                    loop.run_until_complete(send_async(now, frame_image, detected_rects))
                #print("score = " + str(detected_rects[0][1]))
                #save_face_crops(detected_rects, frame_image)
            try:
                l_frame=l_frame.next
            except StopIteration:
                print("ERROR at collecting frame")
                break

        return Gst.PadProbeReturn.OK


    def create_pipeline(self):
        # Convert command to code:
        # command = gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920, height=1080, framerate=30/1, format=NV12' ! nvvidconv flip-method=2 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! ximagesink -e
        # VERY IMPORTANT: since deepstream is used, we need to use nv methods instead of regular methods

        # Create gstreamer elements
        # Create Pipeline element that will form a connection of other elements

        # Standard GStreamer initialization
        GObject.threads_init()
        Gst.init(None)

        print("Creating Pipeline \n ")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
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

        # Use nvinfer to run inferencing on camera's output,
        # behaviour of inferencing is set through config file
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write(" Unable to create pgie \n")

        # Use convertor to convert from NV12 to RGBA as required by nvosd
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")

        # Use a filter to convert to RGBA, since we're not using nvosd anymore
        caps_rgba = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter_rgba = Gst.ElementFactory.make("capsfilter", "filter1")
        if not filter_rgba:
            sys.stderr.write(" Unable to create filter_rgba \n")
        filter_rgba.set_property("caps", caps_rgba)

        if (self.show_stream):
            # Create OSD to draw on the converted RGBA buffer
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            if not nvosd:
                sys.stderr.write(" Unable to create nvosd \n")

            # Finally render the osd output
            if is_aarch64():
                transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")
        else:
            sink = Gst.ElementFactory.make("fakesink", "fakesink")

        #caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=21/1, format=NV12"))
        caps_input.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12"))
#        caps_input.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1, format=NV12"))
        caps_convertres.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=960, height=544"))
        source.set_property('bufapi-version', 1)
        streammux.set_property('width', 960)
        streammux.set_property('height', 544)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)
        pgie.set_property('config-file-path', parameters.CONFIG_FILE_PATH)
        if (self.show_stream):
            # Set sync = false to avoid late frame drops at the display-sink
            sink.set_property('sync', False)

        print("Adding elements to Pipeline \n")
        self.pipeline.add(source)
        self.pipeline.add(caps_input)
        self.pipeline.add(vidconvsrc)
        self.pipeline.add(nvvidconvsrc)
        self.pipeline.add(caps_convertres)
        self.pipeline.add(streammux)
        self.pipeline.add(pgie)
        self.pipeline.add(nvvidconv)
        self.pipeline.add(filter_rgba)
        if self.show_stream:
            self.pipeline.add(nvosd)
        self.pipeline.add(sink)
        if is_aarch64() and self.show_stream:
            self.pipeline.add(transform)

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
        nvvidconv.link(filter_rgba)
        if is_aarch64() and self.show_stream:
            filter_rgba.link(nvosd)
            nvosd.link(transform)
            transform.link(sink)
        else:
            filter_rgba.link(sink)

        # create an event loop and feed gstreamer bus mesages to it
        self.loop = GObject.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.
        osdsinkpad = sink.get_static_pad("sink")
        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")

        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

    def process_buffer(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while(True):
           if not self.queue.empty():
                item = self.queue.get()
                if item is None:
                   break
                now, image, detected_rects = item
                print("Current size of queue is " + str(self.queue.qsize()))
                before = round(time.time() * 1000)
#                send(now, image, detected_rects)
#                loop = asyncio.get_event_loop()
                loop.run_until_complete(send_async(now, image, detected_rects))
                now = round(time.time() * 1000)
                diff = now - before
                print("API call took " + str(diff) + " milliseconds")
           else:
                time.sleep(0.01)



    def start_workers(self, worker_pool=6):
        threads=[]
        for i in range(worker_pool):
             th = threading.Thread(target = self.process_buffer)
             th.start()
             threads.append(th)
        return threads

    def stop_workers(self, threads):
        for i in threads:
             self.queue.put(None)
        for t in threads:
             t.join()

    def start_camera(self):
        print("Starting frame queue \n")
#        workers = self.start_workers(worker_pool=8)
        self.create_pipeline()
        workers = self.start_workers()
        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except :
            pass

        # cleanup
        print("Exiting app")
        self.pipeline.set_state(Gst.State.NULL)
        self.queue.put(None)
        self.stop_workers(workers)
