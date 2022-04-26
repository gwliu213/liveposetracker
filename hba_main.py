# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pyopenpose as op
from redis import Redis
from structs import SyncedQueue
from io import BytesIO
import numpy as np
import logging
from sklearn.cluster import KMeans
from scipy.spatial import distance
import time
from PIL import Image
import json
from hba_tracklet_analyzer import HBATrackletAnalyzer
import pandas as pd
from deep_sort.iou_matching import iou_cost
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker as DeepTracker
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.linear_assignment import min_cost_matching
from deep_sort.detection import Detection as ddet
from tools import generate_detections as gdet
from utils import poses2boxes

import Constants
TRACKLET_SIZE = 27
stream_pose_tracker = "POSE_TRACKER"
INPUT_STREAM = 'Q_FOR_TKL_Person'
#INPUT_STREAM = 'SIM_TRACKLETS'
stream_pose_ROI = 'POSE_ROI' # pose keypoints on ROI
stream_pose = 'POSE_OUTPUT'# pose keypoints on image
stream_image_wh = 'Image_wh'
stream_image_ROI = 'Image_ROI'
tklH = SyncedQueue(maxlen=1000)
frmH = SyncedQueue(maxlen=1000)
combH = SyncedQueue(maxlen=1000)

# config = {
#     'host': 'adam.dfreve-redis',
#     'port': 6379,
#     'password': 'dfreve2019'
# }
class Tracker():
    def __init__(self, debug = False):
        #from openpose import *
        params = dict()
        params["model_folder"] = Constants.openpose_modelfolder
        params["net_resolution"] = "-1x320"



        max_cosine_distance = Constants.max_cosine_distance
        nn_budget = Constants.nn_budget
        self.nms_max_overlap = Constants.nms_max_overlap
        max_age = Constants.max_age
        n_init = Constants.n_init

        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepTracker(metric, max_age = max_age,n_init= n_init)

        Constants.SCREEN_HEIGHT = 640
        Constants.SCREEN_WIDTH = 480




def pose_example():
    try:
        r = Redis( config['host'], 
                   config['port'], 
                   password = config['password'])
        r.ping()
        data = r.xrange(INPUT_STREAM)
        #print(data)
    except Exception as e:
        print(e)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default="../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()
        params = dict()
        params["model_folder"] = "../../models/"
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(args[0].image_path)
        
        #print(imageToProcess)
        #print(type(imageToProcess))
        #print(imageToProcess.shape)
        destRGB = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2RGB)
        #print(destRGB)
        #print(type(destRGB))
        #print(destRGB.shape)
        # B = imageToProcess[:,:,0] 
        # G = imageToProcess[:,:,1] 
        # R = imageToProcess[:,:,2]
        # lst = [R,G,B]
        # np.asarray(lst)
        # print(imageToProcess)
        # print(type(imageToProcess))
        # print(imageToProcess.shape)

        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        # Display Image
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        cv2.imwrite("result_body.jpg", datum.cvOutputData)
        #cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        #cv2.waitKey(0)
    except Exception as e:
        print(e)
        sys.exit(-1)

def connect(config):
    try:
        r = Redis( config['host'], config['port'], password = config['password'])
        r.ping()
    except Exception as e:
        print('Unable to ping Redis')
        print(e)
    return r

def getObjects(r, stream, prev_ID=None):
    if not prev_ID:
        while True:
            bulk_data = r.xrange(stream)
            if bulk_data:
                break
    else:
        bulk_data = []
        while not bulk_data:
            bulk_data = r.xrange(stream, prev_ID)
            #print(bulk_data[-1][0])
    return bulk_data, bulk_data[-1][0]


def getVid(r, stream='Q_TO_VID', prev_ID=None):
    # get the image frame bulk data
    bulk_images, prev_ID = getObjects(r, stream, prev_ID)
    # split frame into imgId and frame
    dict_images = splitImageIDImgs(bulk_images)
    # return dictionary of imgId, frame and prev_ID
    return dict_images, prev_ID

def splitImageIDImgs(bulk_images):
    global frmH
    dictImages = {}
    for imgStr in bulk_images:
        imgID, img = imgStr[1][b'frame'].split(b":", 1)
        #img = np.load(BytesIO(img))
        imgID = imgID.decode()
        dictImages[imgID] = img
        frmH.put(img, imgID)
    return dictImages

def splitTKL(bulk_tkls):
    dictImgIDToPpl = {}
    for tkl in bulk_tkls:
        imgID, l_tkls = tkl[1][b'tracklets'].split(b':')
        imgID = imgID.split(b'.')[1].decode()
        tklH.put(l_tkls, imgID)
        dictImgIDToPpl[imgID] = l_tkls
    return dictImgIDToPpl

def getTKLFrame(r, stream='Q_FOR_TKL_Person', prev_ID=None):
    bulk_tkls, prev_ID = getObjects(r, stream, prev_ID)
    dict_tkls = splitTKL(bulk_tkls)
    return dict_tkls, prev_ID

def extractImage(imageID: str):
    imgStr = frmH.get(imageID)
    img = np.load(BytesIO(imgStr))
    img = cv2.imdecode(img, 1)
    #img = np.array(img, order='C')
    return img

def extractTracklets(imageID: str) -> np.ndarray:
    # trackletStr format: [name, tracklet, (optional) transformedTracklet]
    tracklets = []
    transformedTracklets = []
    trackletStr = tklH.get(imageID)
    #logging.info(trackletStr.decode())
    tracklets += json.loads(trackletStr.decode())
    tracklets = np.array(tracklets, dtype=np.int32).reshape(-1, TRACKLET_SIZE)
    return tracklets

def compare(r):
    #logging.info(f'coming here 1')
    count = 0
    m_count = 0
    allMatched = True
    ret_val = 0
    for i in range(len(frmH)):
        img_id = frmH[0]
        if not tklH.contains(img_id):
            #    logging.info(f'{img_id} in frmH but not in tklH')
            #logging.info(img_id)
            frmH.get(img_id)
            count += 1
            allMatched = False
            ret_val = -1
        else:
            m_count += 1
            image = extractImage(img_id) 
            #cv2.imwrite('./'+NAME+'_image/'+str(img_id)+'.png', image)
            tracklets = extractTracklets(img_id)
            combH.put((image, tracklets), img_id)
    #logging.info(f'matched_count = {m_count} , unmatched = {count}, num_imgs and num_tkls={len(combH)}')

    return ret_val
def closest_node(node, cluster):
    closest_index = distance.cdist([node], cluster).argmin()
    return closest_index
def analyze_keypoints(keyBodyPoints,model):
    matrix = { "xc": [], "yc": [],"BB_pixels":[] }
    # BB threshold = 4000 pixels wrt. 768*432=331776
    if type(keyBodyPoints) == list:
        points = []
        for person in keyBodyPoints:
            # for i in range(len(person)):
            #     matrix[i].append(person[i])
            # person_sum = [sum(i) for i in zip(*person)]
            # person_c= (person_sum[0]/25,person_sum[1]/25)
            matrix["xc"].append(person[1][0])
            matrix["yc"].append(person[1][1])
            gen = [x[0:-1] for x in person if x[2] > 0]
            person_sum = [sum(i) for i in zip(*gen)]
            person_c = [person[1][0], person[1][1]]
            # person_c = [person_sum[0] / len(gen), person_sum[1] / len(gen)]
            points.append(person_c)
            # for x in person:
            #     points.append(x[0:-1])
            xmin = min(gen, key=lambda x: x[0])[0]
            xmax = max(gen, key=lambda x: x[0])[0]
            ymin = min(gen, key=lambda x: x[1])[1]
            ymax = max(gen, key=lambda x: x[1])[1]
            BB_pixels = int((xmax-xmin)*(ymax-ymin))
            matrix["BB_pixels"].append(BB_pixels)
        data = np.array(points)
        model.fit(data)
        labels = model.labels_.tolist()
        most_common = max(labels, key=labels.count)
        cluster_c = model.cluster_centers_[most_common].tolist()
        nearest_person_idx = closest_node(cluster_c, points)
        # print(nearest_person_idx)
        center_BB_pix = matrix["BB_pixels"][nearest_person_idx]
        # print('center_BB_pix=',center_BB_pix)
        return (cluster_c, center_BB_pix)
    else:
        return False

def centerFromPose(keyBodyPoints):
    center_x = []
    center_y = []
    w = []
    h = []
    if type(keyBodyPoints) == list:
        for person in keyBodyPoints:
            gen = [x[0:-1] for x in person if x[2] > 0]
            if len(gen) > 0:
                xmin = min(gen, key=lambda x: x[0])[0]
                xmax = max(gen, key=lambda x: x[0])[0]
                ymin = min(gen, key=lambda x: x[1])[1]
                ymax = max(gen, key=lambda x: x[1])[1]
                h.append(ymax - ymin)
                w.append(xmax - xmin)
                person_sum = [sum(i) for i in zip(*gen)]
                center_x.append(person_sum[0] / len(gen))
                center_y.append(person_sum[1] / len(gen))
        return center_x,center_y,w,h
def get_poses_imgW(r,tracker,pose_model, redis_stream,videoWriter):
    # print('Inside pose')
    ppl = []
    len1 = len(combH)
    # stream_2 = 'POSE_OUTPUT'
    stream_2 = redis_stream
    model = KMeans(n_clusters=2)
    # buffer = BytesIO()
    ### Get the image
    try:
        for i in range(len1):
            img_id = combH[0]
            person_copy = []
            image_wh = []
            cluster = True
            currentFrame, tkl = combH.get(img_id)
            h,w,c = currentFrame.shape

            keyBodyPoints = pose_model.get_bodykepoints(currentFrame)
            # #################### pose keypoint on whole image ####################
            image_wh.append(str(img_id))
            image_wh.append(keyBodyPoints.tolist())
            id = r.xadd(stream_pose, {'person': str(image_wh)})

            c_x,c_y,BB_w,BB_h=centerFromPose(keyBodyPoints.tolist())

            #################### whole image stream ####################
            currentFrame = pose_model.datum.cvOutputData
            #################################tracker#################################################

            # Doesn't use keypoint confidence
            poses = np.array(keyBodyPoints)[:, :, :2]
            # Get containing box for each seen body
            boxes = poses2boxes(poses)

            boxes_xywh = [[1 if (x1-(x2-x1)/10)<=1 else (x1-(x2-x1)/10) ,
                           1 if (y1-(y2-y1)/10)<=1 else (y1-(y2-y1)/10),
                           w-1 if (x1+(x2-x1)*1.2)>=w-1 else (x2 - x1)*1.2,
                           h-1 if (y1+(y2-y1)*1.2)>=h-1 else (y2 - y1)*1.2 ]
                          for [x1, y1, x2, y2] in boxes]
            # print("boxes before", len(boxes_xywh))
            # average_BB=np.mean(boxes_xywh, axis=0)
            # average_size =average_BB[2]*average_BB[3]
            # print("mean",average_size)
            features = tracker.encoder(currentFrame, boxes_xywh)
            # print(features)

            nonempty = lambda xywh: xywh[2] != 0 and xywh[3] != 0
            detections = [Detection(bbox, 1.0, feature, pose) for bbox, feature, pose in
                          zip(boxes_xywh, features, poses) if nonempty(bbox)]
            # Run non-maxima suppression.
            boxes_det = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes_det, tracker.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # Call the tracker
            tracker.tracker.predict()
            tracker.tracker.update(currentFrame, detections)
            posetrack = []
            poses_perimage = []
            IDs = []
            for track in tracker.tracker.tracks:
                color = None
                if not track.is_confirmed():
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)
                bbox = track.to_tlbr()

                # print("bbox:",bbox)
                # print("Body keypoints:")
                # print("type of pose", type(track.last_seen_detection.pose))
                poses_perimage.append(track.last_seen_detection.pose.tolist())
                IDs.append(int(track.track_id))
                # print("type of ID", type(track.track_id))

                # print("ID=",poses_perimage)
                # print(track.last_seen_detection.pose)
                # if int(bbox[2]-bbox[0])*int(bbox[3]-bbox[1])<= 5000:
                #     print("bbox size=", int(bbox[2]-bbox[0])*int(bbox[3]-bbox[1]))
                cv2.rectangle(currentFrame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(currentFrame, "%s" % (track.track_id),
                            (int(bbox[0]), int(bbox[1]) - 20), 0, 5e-3 * 100, (0, 255, 0), 2)
            posetrack.append(str(img_id))
            posetrack.append(poses_perimage)
            posetrack.append(IDs)
            id = r.xadd(stream_pose_tracker, {'posetracker': str(posetrack)})
            # print("bbox after =", len(tracker.tracker.tracks))

            videoWriter.write(currentFrame)
            ###################################################################################

            status, image_wh = cv2.imencode('.jpg', currentFrame)
            sio = BytesIO()
            np.save(sio, image_wh)
            imgIO_wh = sio.getvalue()
            img_wh = imgIO_wh
            #  whole image
            id2 = r.xadd(stream_image_wh, {'Image_id': str(img_id),
                                       'Image': img_wh})

    except Exception as e:
        print(str(e))
    return

def get_poses(r, pose_model,redis_stream):
    #print('Inside pose')
    ppl = []
    len1 = len(combH)
    #stream_2 = 'POSE_OUTPUT'
    stream_2 = redis_stream 
    #buffer = BytesIO()
    ### Get the image
    try:
        for i in range(len1):
            img_id = combH[0]
            img, tkl = combH.get(img_id)
            print("tkl=", tkl)
            ppl = getTrackletsInImage(img, tkl, img_id)
            #buffer = BytesIO()
            #print(type(img))
            
            for person in ppl:
                print(person[2])
                destRGB = cv2.cvtColor(person[2], cv2.COLOR_BGR2RGB)
                keyBodyPoints = pose_model.get_bodykepoints( destRGB )
                np_pose_array = pose_model.datum.cvOutputData
                #if save_poses: 
                #cv2.imwrite('/data/' + str(person[0]) + '_' + str(person[1]) +'.jpeg', pose_model.datum.cvOutputData)  

                person_copy = person[:]
                person_copy.append( keyBodyPoints.tolist() )
                person_copy.append( str(img_id) )
                person_copy.append( np_pose_array.tolist() )
                del person_copy[2]
                
                id = r.xadd( stream_2, {'person': str(person_copy)} )

            # id2 = r.xadd( 'Image_Stream', {'Image_id': str(img_id),
            #                                'Image':  img.tobytes() } )
            
            _ , pose_image = cv2.imencode('.jpg', np_pose_array)
            sio = BytesIO()
            np.save(sio, pose_image)
            pose_imgIO = sio.getvalue()
            pose_img = pose_imgIO
            id2 = r.push('Image_pose', {'Image_id': str(img_id),
                                          'Image': pose_img})
            print('Image added')

            status, image = cv2.imencode('.jpg', img)
            sio = BytesIO()
            np.save(sio, image)
            imgIO = sio.getvalue()
            img = imgIO
            id2 = r.xadd('Image_Stream', {'Image_id': str(img_id),
                                          'Image': img})

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    return ppl

def getTrackletsInImage(img, tkl, img_id):
    '''
        Fall detection algorithm 
    '''
    ppl = []
    exception_counter = 0
    for n in range(tkl.shape[0]):
        x, y, w, h = tkl[n][2:6]   # (x,y) is bot left of bbox
        name = str(tkl[n][1])[-6:]
        name = hex(int(name))[2:7] #remove 0x prefix
        xmin = int(round(x - (w /2.)));  xmax = int(round(x + (w /2.)))
        ymin = int(round(y - (h /2.)));  ymax = int(round(y + (h /2.)))
        print(w,h)
        if w + 10 > h: 
            ppl.append( [img_id, name, img[ymin:ymax,xmin:xmax], (xmin,ymin), (xmax,ymax), 1] )
        else:
            ppl.append( [img_id, name, img[ymin:ymax,xmin:xmax],(xmin,ymin), (xmax,ymax), 0] )
    return ppl 

class poseModel:
    def __init__(self):
        self.params = dict()
        self.params["model_folder"] = "../../models/"
        self.model = self.initModel(self.params)
        
    def initModel(self, params):
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        return opWrapper

    def get_bodykepoints(self, img):
        self.datum = op.Datum()
        self.datum.cvInputData = img
        self.model.emplaceAndPop([self.datum])

        return self.datum.poseKeypoints

def processH(r, tracker,pose_model,redis_stream):
    global people
    fps = 10
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video_title = os.getenv('video_title','my_pose_video')
    vpath = '/data/'+video_title+'.avi'
    roi_vpath = '/data/'+video_title+'_ROI.avi'
    #print('Writing '+ video_title + ' to : ' + str(os.getcwd()) )
    videoWriter = cv2.VideoWriter(vpath,fourcc,fps,(768,432))

    prev_ID = None
    prev_tID = None
    while True:
        _, prev_ID = getVid(r, prev_ID=prev_ID)
        try:
            _, prev_tID = getTKLFrame(r, prev_ID=prev_tID)

        except Exception as e:
            logging.error(e)
        ret = compare(r)
        if ret == 0:
            get_poses_imgW(r, tracker,pose_model,redis_stream,videoWriter)
    videoWriter.release()


def process(r, pose_model,redis_stream):
    global people
    prev_ID = None
    prev_tID = None
    while True:
        _, prev_ID = getVid(r, prev_ID=prev_ID)
        #print('video frames collected.')
        try:
            _, prev_tID = getTKLFrame(r, prev_ID=prev_tID)
            #print('Tracklets collected')
        except Exception as e:
            logging.error(e)
        ret = compare(r)
        if ret == 0:
            people = get_poses(r, pose_model,redis_stream)

def main(config,redis_stream,version):
    r = connect(config)
    pose_model = poseModel()
    tracker = Tracker()
    
    if version == 'adam': process(r, pose_model,redis_stream)
    if version == 'gaowen': processH(r, tracker,pose_model,redis_stream)#gaowen

if __name__ == "__main__":
    
    #global save_poses

    defaultLoggingLevel=logging.INFO 
    loggingFORMAT = "%(asctime)20s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(format=loggingFORMAT, level=logging.INFO)
    
    redis_host = os.getenv('redis_host','localhost')
    redis_port = int(os.getenv('redis_port',6379))
    redis_stream = os.getenv('redis_stream','POSE_OUTPUT')
    #save_poses = os.getenv('save_poses', True)

    version = os.getenv('version', 'gaowen')
    

    config = {
        'host': redis_host,
        'port': redis_port,
        'password': 'dfreve2019'
    }

    print('Config loaded.') 
    print('Begining Human Behavior Analytics')    
    main(config,redis_stream,version)

    # **** BELOW IS OLD CODE THAT COULD BE USED LATER *** 

    # EXAMPLE = False
    # POSE = True
    # ANALYTICS = False

    # defaultLoggingLevel=logging.INFO 
    # loggingFORMAT = "%(asctime)20s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # logging.basicConfig(format=loggingFORMAT, level=logging.INFO)

    # if ANALYTICS  == True:
        
    #     #stream_prefix = "Q_FOR_TKL_" #param
    #     #classid = 'Person' #param
    #     stream_prefix = 'SIM_TRACKLETS'
    #     classid = ''
    #     radius = 100 #param
    #     prev_ID = None #param
    #     threshold = 30 #param
    #     maskWindow=10000 #param
    #     df_raw = pd.DataFrame()
    #     # initialize
    #     print(stream_prefix)
    #     hbata = HBATrackletAnalyzer(host=config['host'], port=config['port'], password = config['password'], 
    #                                 stream_prefix=stream_prefix, classid=classid, threshold=threshold)
    #     # start the listening loop
    #     hbata.startListening(prev_ID=prev_ID, radius=radius, df_raw=df_raw, displayStats=True, maskWindow=maskWindow)

    # if EXAMPLE == True:
    #     pose_example()

    