import os
import sys
import cv2
import numpy as np
import math
import onnxruntime
import argparse

def sample_target(im, target_bb, search_area_factor, output_sz):
    x, y, w, h = target_bb
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1: raise Exception('Too small bounding box.')
    x1 = round(x + 0.5*w - crop_sz*0.5)
    x2 = x1 + crop_sz
    y1 = round(y + 0.5*h - crop_sz*0.5)
    y2 = y1 + crop_sz
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    resize_factor = output_sz / crop_sz
    im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
    return im_crop_padded, resize_factor

def process(img_arr):
    img_arr = np.concatenate([img_arr, img_arr], axis=-1)
    img_tensor = img_arr.astype(np.float32).transpose(2,0,1)
    mean = np.array([0.485,0.456,0.406,0.485,0.456,0.406], dtype=np.float32).reshape(6,1,1)
    std = np.array([0.229,0.224,0.225,0.229,0.224,0.225], dtype=np.float32).reshape(6,1,1)
    return ((img_tensor/255.0)-mean)/std

def transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz):
    box_extract_center = np.array(box_extract[0:2]) + 0.5*np.array(box_extract[2:4])
    box_in_center = np.array(box_in[0:2]) + 0.5*np.array(box_in[2:4])
    box_out_center = (crop_sz-1)/2 + (box_in_center - box_extract_center)*resize_factor
    box_out_wh = np.array(box_in[2:4])*resize_factor
    box_out = np.concatenate((box_out_center - 0.5*box_out_wh, box_out_wh))
    return box_out/(crop_sz-1)

def clip_box(box, H, W, margin):
    x1,y1,w,h = box
    x2,y2 = x1+w, y1+h
    x1 = min(max(0,x1), W-margin)
    x2 = min(max(margin,x2), W)
    y1 = min(max(0,y1), H-margin)
    y2 = min(max(margin,y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1,y1,w,h]

def hann1d(sz, centered=True):
    if centered: return 0.5*(1-np.cos((2*math.pi/(sz+1))*np.arange(1,sz+1,dtype=np.float32)))
def hann2d(sz, centered=True):
    h1_0 = hann1d(sz[0], centered).reshape(1,1,-1,1)
    h1_1 = hann1d(sz[1], centered).reshape(1,1,1,-1)
    return h1_0 * h1_1

def cal_bbox(score_map_ctr, size_map, offset_map, feat_sz):
    score_map_flat = score_map_ctr.flatten()
    idx = np.argmax(score_map_flat)
    max_score = score_map_flat[idx]
    idx_y = idx//feat_sz
    idx_x = idx%feat_sz
    size = size_map.reshape(2,-1)[:,idx]
    offset = offset_map.reshape(2,-1)[:,idx]
    bbox = np.array([(idx_x+offset[0])/feat_sz, (idx_y+offset[1])/feat_sz, size[0], size[1]], dtype=np.float32)
    return bbox, max_score

def map_box_back(pred_box, state, resize_factor, search_size):
    cx_prev, cy_prev = state[0]+0.5*state[2], state[1]+0.5*state[3]
    cx,cy,w,h = pred_box
    half_side = 0.5*search_size/resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real-0.5*w, cy_real-0.5*h, w, h]

def get_model_config(onnx_path):
    name = os.path.basename(onnx_path).lower()
    if "384" in name: return 384,192
    else: return 224,112

# ------------------- 防抖函数（支持强度+半径） -------------------
def stabilize_frame(prev_gray, curr_gray, strength=0.7, radius=30):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.3, minDistance=radius)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    idx = np.where(status==1)[0]
    if len(idx) < 10: return np.eye(2,3)
    m, _ = cv2.estimateAffinePartial2D(prev_pts[idx], curr_pts[idx])
    if m is None: return np.eye(2,3)
    m[0,2] *= strength
    m[1,2] *= strength
    return m

def main():
    parser = argparse.ArgumentParser(description="SUTrack Demo")
    parser.add_argument("--video", type=str, default='./videos/demo_video_bag.mp4')
    parser.add_argument("--model", type=str, default='./onnx_models/sutrack_b384.onnx')
    parser.add_argument("--stabilize", action="store_true", help="开启目标锁定防抖")
    parser.add_argument("--stabilize_strength", type=float, default=0.7, help="防抖强度 0~1")
    parser.add_argument("--stabilize_radius", type=int, default=30, help="特征点最小半径")
    args = parser.parse_args()

    search_size, template_size = get_model_config(args.model)
    ort_session = onnxruntime.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(args.video)
    ret, first = cap.read()
    if not ret: return

    # 视频保存
    H, W = first.shape[:2]
    fps = 25
    out = cv2.VideoWriter("track_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("SUTrack", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("SUTrack", first, False, True)
    if roi[2]<=0: return

    template_factor=2.0
    search_factor=4.0
    num_templates=2
    update_intervals=25
    update_threshold=0.7
    feat_sz=search_size//16
    miss_threshold=0.45
    window = hann2d((feat_sz, feat_sz), True)

    state = list(roi)
    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    z_patch, z_scale = sample_target(first_rgb, state, template_factor, template_size)
    template = process(z_patch)
    t_list = [template]*num_templates
    anno = transform_image_to_crop(state, state, z_scale, template_size)
    anno_list = [anno]*num_templates

    last_m = np.eye(2,3)

    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # ------------------- 防抖开关 -------------------
        if args.stabilize:
            m = stabilize_frame(prev_gray, curr_gray, args.stabilize_strength, args.stabilize_radius)
            display = cv2.warpAffine(frame, m, (W,H))
            curr_gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

        # ------------------- 追踪 -------------------
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        x_patch, x_scale = sample_target(rgb, state, search_factor, search_size)
        search = process(x_patch)

        ort_inputs = {
            "template": np.stack(t_list,0)[None,...],
            "search": search[None,None,...],
            "template_anno": np.stack(anno_list,0)[None,...].astype(np.float32)
        }
        score, size, offset = ort_session.run(None, ort_inputs)
        box, conf = cal_bbox(score*window, size, offset, feat_sz)
        box = (box * search_size / x_scale).tolist()
        state = clip_box(map_box_back(box, state, x_scale, search_size), H,W,10)

        # ------------------- 模板更新 -------------------
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % update_intervals ==0 and conf>update_threshold:
            z_patch, z_scale = sample_target(rgb, state, template_factor, template_size)
            t_list.append(process(z_patch))
            if len(t_list)>num_templates: t_list.pop(1)
            anno = transform_image_to_crop(state,state,z_scale,template_size)
            anno_list.append(anno)
            if len(anno_list)>num_templates: anno_list.pop(1)

        # ------------------- 绘制 -------------------
        x,y,w,h = [int(v) for v in state]
        color = (0,255,0) if conf>miss_threshold else (0,0,255)
        cv2.rectangle(display, (x,y), (x+w,y+h), color, 2)
        cv2.putText(display, f"{conf:.2f}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        # ------------------- 保存视频 -------------------
        out.write(display)
        prev_gray = curr_gray.copy()
        cv2.imshow("SUTrack", display)
        if cv2.waitKey(1)&0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
