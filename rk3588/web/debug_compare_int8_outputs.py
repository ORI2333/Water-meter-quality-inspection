
import sys, os, numpy as np, cv2, time
from rknnlite.api import RKNNLite
sys.path.insert(0, '/home/demo/water_meter/code')
import hdmi_yolo11_pose_detect as pose

def run(model):
    r=RKNNLite(); assert r.load_rknn(model)==0; assert r.init_runtime()==0
    frame=cv2.imread('/home/demo/water_meter/pose_known_val.jpg')
    inp, ratio, px, py=pose.prepare_input(frame, 640, 640, 'rgb', 'nhwc', 'uint8')
    out=r.inference(inputs=[inp], data_type='uint8', data_format='nhwc')
    arr=np.asarray(out[0])
    while arr.ndim>2 and arr.shape[0]==1: arr=arr[0]
    if arr.shape[0] < arr.shape[1]: arr=arr.T
    print('\nMODEL', os.path.basename(model), 'shape', arr.shape, 'minmax', arr.min(), arr.max())
    cls=arr[:,4:8]
    print('cls raw min/max/mean', cls.min(), cls.max(), cls.mean())
    sig=1/(1+np.exp(-np.clip(cls,-50,50)))
    print('cls sigmoid min/max/mean', sig.min(), sig.max(), sig.mean())
    maxs=sig.max(axis=1); ids=sig.argmax(axis=1)
    top=np.argsort(-maxs)[:12]
    for i in top:
        print('top', int(i), 'id', int(ids[i]), 'score', float(maxs[i]), 'rawcls', cls[i].round(3).tolist(), 'xywh', arr[i,:4].round(1).tolist(), 'kptscore', arr[i,[10,13]].round(3).tolist())
    for conf in [0.001,0.005,0.01,0.02,0.05,0.1,0.25]:
        dets=pose.postprocess_pose(out, conf, 0.45, 640, 640)
        print('conf',conf,'dets',len(dets), [(d.cls_id, round(d.score,4)) for d in dets[:8]])
    r.release()

for m in [
'/home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn',
'/home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_raw_normal.rknn',
'/home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_lb_normal.rknn',
'/home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_lb_mmse.rknn']:
    run(m)
