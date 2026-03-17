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

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    
    resize_factor = output_sz / crop_sz
    im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
    return im_crop_padded, resize_factor

def process(img_arr):
    # img_arr is (H, W, 3)
    img_arr = np.concatenate([img_arr, img_arr], axis=-1) # (H, W, 6)
    img_tensor = img_arr.astype(np.float32).transpose(2, 0, 1) # (6, H, W)
    mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32).reshape(6, 1, 1)
    std = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32).reshape(6, 1, 1)
    return ((img_tensor / 255.0) - mean) / std

def transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz):
    box_extract_center = np.array(box_extract[0:2]) + 0.5 * np.array(box_extract[2:4])
    box_in_center = np.array(box_in[0:2]) + 0.5 * np.array(box_in[2:4])
    
    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = np.array(box_in[2:4]) * resize_factor
    
    box_out = np.concatenate((box_out_center - 0.5 * box_out_wh, box_out_wh))
    return box_out / (crop_sz - 1)

def clip_box(box, H, W, margin):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]

def hann1d(sz, centered=True):
    if centered:
        return 0.5 * (1 - np.cos((2 * math.pi / (sz + 1)) * np.arange(1, sz + 1, dtype=np.float32)))

def hann2d(sz, centered=True):
    h1_0 = hann1d(sz[0], centered).reshape(1, 1, -1, 1)
    h1_1 = hann1d(sz[1], centered).reshape(1, 1, 1, -1)
    return h1_0 * h1_1

def cal_bbox(score_map_ctr, size_map, offset_map, feat_sz):
    score_map_flat = score_map_ctr.flatten()
    idx = np.argmax(score_map_flat)
    max_score = score_map_flat[idx]
    
    idx_y = idx // feat_sz
    idx_x = idx % feat_sz
    
    size_map_flat = size_map.reshape(2, -1)
    size = size_map_flat[:, idx]
    
    offset_map_flat = offset_map.reshape(2, -1)
    offset = offset_map_flat[:, idx]
    
    bbox = np.array([
        (idx_x + offset[0]) / feat_sz,
        (idx_y + offset[1]) / feat_sz,
        size[0],
        size[1]
    ], dtype=np.float32)
    return bbox, max_score


def map_box_back(pred_box, state, resize_factor, search_size):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def get_model_config(onnx_path):
    filename = os.path.basename(onnx_path).lower()

    if "384" in filename:
        search_size = 384
        template_size = 192
    else:
        search_size = 224
        template_size = 112
        
    print(f"Load onnx model: {filename}")
    print(f"Search Size = {search_size}, Template Size = {template_size}")
    return search_size, template_size

def main():
    parser = argparse.ArgumentParser(description="SUTrack ONNX Inference")
    parser.add_argument("--video", type=str, default='./demo_video_bag.mp4', help="Video file path")
    parser.add_argument("--model", type=str, default='./sutrack_b384.onnx', help="ONNX model path")
    args = parser.parse_args()

    video_path = args.video
    onnx_path = args.model

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(onnx_path):
        print(f"Error: Model file not found at {onnx_path}")
        return

    search_size, template_size = get_model_config(onnx_path)

    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return

    window_name = 'SUTrack ONNX Inference'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    roi_bbox = cv2.selectROI(window_name, first_frame, fromCenter=False, showCrosshair=True)
    if roi_bbox[2] <= 0 or roi_bbox[3] <= 0: return

    template_factor = 2.0
    search_factor = 4.0
    num_templates = 2
    update_intervals = 25
    update_threshold = 0.7
    feat_sz = search_size // 16

    output_window = hann2d((feat_sz, feat_sz), centered=True)

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    state = list(roi_bbox)

    z_patch_arr, resize_factor = sample_target(first_frame_rgb, state, template_factor, output_sz=template_size)
    template = process(z_patch_arr)
    
    template_list = [template] * num_templates
    prev_box_crop = transform_image_to_crop(state, state, resize_factor, template_size)
    template_anno_list = [prev_box_crop] * num_templates 
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_id += 1
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        x_patch_arr, resize_factor = sample_target(frame_rgb, state, search_factor, output_sz=search_size)
        search = process(x_patch_arr)
        
        template_input = np.stack(template_list, axis=0)[np.newaxis, ...] 
        search_input = np.stack([search], axis=0)[np.newaxis, ...] 
        template_anno_input = np.stack(template_anno_list, axis=0)[np.newaxis, ...].astype(np.float32) 
        
        ort_inputs = {'template': template_input, 'search': search_input, 'template_anno': template_anno_input}
        score_map, size_map, offset_map = ort_session.run(None, ort_inputs)
        
        response = score_map * output_window
        pred_box, conf_score = cal_bbox(response, size_map, offset_map, feat_sz)
        
        pred_box = (pred_box * search_size / resize_factor).tolist()  
        state = clip_box(map_box_back(pred_box, state, resize_factor, search_size), H, W, margin=10)

        if frame_id % update_intervals == 0 and conf_score > update_threshold:
            z_patch_arr, resize_factor = sample_target(frame_rgb, state, template_factor, output_sz=template_size)
            template = process(z_patch_arr)
            template_list.append(template)
            if len(template_list) > num_templates: template_list.pop(1)
            
            prev_box_crop = transform_image_to_crop(state, state, resize_factor, template_size)
            template_anno_list.append(prev_box_crop)
            if len(template_anno_list) > num_templates: template_anno_list.pop(1)

        res_box = [int(i) for i in state]
        cv2.rectangle(frame, (res_box[0], res_box[1]), (res_box[0]+res_box[2], res_box[1]+res_box[3]), (0, 255, 0), 3)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
