
"""
record.py
---------------

Main Function for recording a video sequence into cad (color-aligned-to-depth) 
images and depth images


"""

# record for 30s after a 5s count down
# or exit the recording earlier by pressing q

RECORD_LENGTH = 30

import png
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import pyrealsense as pyrs
import time
import os
import sys
from pyrealsense.constants import rs_option
# from config.DataAcquisitionParameters import DEPTH_THRESH

def make_directories(folder):
    if not os.path.exists(folder+"cad/"):
        os.makedirs(folder+"cad/")
    if not os.path.exists(folder+"depth/"):
        os.makedirs(folder+"depth/")
    if not os.path.exists(folder+"annotations/"):
        os.makedirs(folder+"annotations/")
    if not os.path.exists(folder+"mask/"):
        os.makedirs(folder+"mask/")
    if not os.path.exists(folder+"results/"):
        os.makedirs(folder+"results/")

        
def print_usage():
    
    print "Usage: record.py <foldername>"
    print "foldername: path where the recorded data should be stored at"
    print "e.g., record.py Data/mug"

def save_color_intrinsics(folder):
    import pyrealsense2 as rs
    import json
    
    with pyrs.Service() as serv:
        with serv.Device() as dev:
            serial = dev.serial
            dev.wait_for_frames()
            c = dev.color
            H,W,_ = c.shape

    dev.stop()
    serv.stop()
   
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # Start pipeline
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Color Intrinsics 
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    pipeline.stop()
    camera_parameters = {'ID': serial, 'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width}

    
    with open(folder+'intrinsics.json', 'w') as fp:
        json.dump(camera_parameters, fp)
     
    

if __name__ == "__main__":
    try:
        folder = sys.argv[1]+"/"
    except:
        print_usage()
        exit()

    make_directories(folder)
    # save_color_intrinsics(folder)
    FileName=0
    
    with pyrs.Service() as serv:
        with serv.Device() as dev:

            # Set frame rate
            cnt = 0
            last = time.time()
            smoothing = 0.9
            fps_smooth = 30
            T_start = time.time()
            while True:
                cnt += 1
                if (cnt % 10) == 0:
                    now = time.time()
                    dt = now - last
                    fps = 10/dt
                    fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
                    last = now

                dev.wait_for_frames()
                c = dev.color
                c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
                d = dev.dac
         
                # Visualize count down
           
                if time.time() -T_start > 5:
                    if FileName%2 == 0:
                        filecad= folder+"cad/%s.jpg" % int(FileName/2)
                        filedepth= folder+"depth/%s.png" % int(FileName/2)
                        cv2.imwrite(filecad,c)
                        with open(filedepth, 'wb') as f:
                            writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                                bitdepth=16, greyscale=True)
                            zgray2list = d.tolist()
                            writer.write(f, zgray2list)
                    FileName+=1
                    
                if FileName >= fps_smooth*RECORD_LENGTH:
                    dev.stop()
                    serv.stop()
                    break

                overlay = np.zeros(c.shape,dtype = np.uint8)
                overlay[(d>0)&(d<0.5/8*65535)] = np.array([255,255,255],
                                                                  dtype = np.uint8) 
                c = cv2.addWeighted(c,0.6,overlay,0.2,0)
                if time.time() -T_start < 5:
                    cv2.putText(c,str(5-int(time.time() -T_start)),(240,320), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4,(0,0,255),2,cv2.LINE_AA)
                if time.time() -T_start > RECORD_LENGTH:
                    cv2.putText(c,str(RECORD_LENGTH+5-int(time.time()-T_start)),(240,320), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4,(0,0,255),2,cv2.LINE_AA)
                cv2.imshow('COLOR IMAGE',c)
                
           
                    
                # press q to quit the program
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    dev.stop()
                    serv.stop()
                    break

    # Release everything if job is finished
    cv2.destroyAllWindows()
