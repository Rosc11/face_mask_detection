
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

from utils import save_face_crops, get_cv2_image, apply_nms
import pyds

class Camera:
    def __init__(self):
        self.pipeline = None
        self.loop = None

    def osd_sink_pad_buffer_probe(self, pad,info,u_data):
        print("Entered buffer probe.")
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
            i=0
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
                i = i + 1
                try:
                    l_obj=l_obj.next
                except StopIteration:
                    break

            #apply nms
            detected_rects = apply_nms(detected_rects, scores)
            print("Frame " + str(frame_number)  + ", " + str(len(detected_rects))+" object(s) detected!")

            #retrieve & save image
            if (len(detected_rects) > 0):
                frame_image = get_cv2_image(gst_buffer, frame_meta)
                #save_face_crops(detected_rects, frame_image)
            try:
                l_frame=l_frame.next
            except StopIteration:
                print("ERROR at collecting frame")
                break
        return Gst.PadProbeReturn.OK


    def create_pipeline(self, show_stream):
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

        # Create OSD to draw on the converted RGBA buffer
        # This is where the probe is placed to draw on images
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")

        # Finally render the osd output
        if is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

        if (show_stream):
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
        if (show_stream):
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
        self.pipeline.add(nvosd)
        self.pipeline.add(sink)
        if is_aarch64():
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
        nvvidconv.link(nvosd)
        if is_aarch64() and show_stream:
            nvosd.link(transform)
            transform.link(sink)
        else:
            nvosd.link(sink)

        # create an event loop and feed gstreamer bus mesages to it
        self.loop = GObject.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect ("message", bus_call, self.loop)

        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")

        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)


    def start_camera(self):
        print("Starting pipeline \n")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except :
            pass

        # cleanup
        print("Exiting app")
        self.pipeline.set_state(Gst.State.NULL)
