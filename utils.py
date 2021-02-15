import os
import cv2
import numpy as np
from nms import nms
import pyds

def save_face_crops(detected_rects, frame_image):
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
        if not os.path.exists('imgs'):
           os.makedirs('imgs')
           cv2.imwrite("imgs/frame_"+str(frame_number)+"_crop_"+str(i)+"_"+label+".jpg",frame_crop)

def get_cv2_image(gst_buffer, frame_meta):
    # the input should be address of buffer and batch_id
    n_frame=pyds.get_nvds_buf_surface(hash(gst_buffer),frame_meta.batch_id)
    #convert python array into numy array format.
    frame_image=np.array(n_frame,copy=True,order='C')
    #covert the array into cv2 default color format
    frame_image=cv2.cvtColor(frame_image,cv2.COLOR_RGBA2BGRA)
    return frame_image

def resize_image_post_request(image, width, height):
    image = cv2.resize(image, (width, height))
    return image

### expects detections in [class_id, x, y, w, h] format. len(scores) should match len(detections)
def apply_nms(detections, scores):
    det_rects = np.array(detections)
    scores = np.array(scores)
    rects = [rect[1:5] for rect in det_rects]
    indices = nms.boxes(rects, scores)
    detected_rects = det_rects[indices]
    scores = scores[indices]
    return detected_rects

### expects detections in [class_id, score, x, y, w, h] format
def apply_nms(detections):
    det_rects = np.array(detections)
    rects = [rect[2:6] for rect in det_rects]
    scores = [rect[1] for rect in det_rects]
    indices = nms.boxes(rects, scores)
    detected_rects = det_rects[indices]
    return detected_rects
