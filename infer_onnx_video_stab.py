import os
import cv2
import numpy as np
import math
import onnxruntime
import argparse

def sample_target(im, target_bb, search_area_factor, output_sz):
    x, y, w, h = target_bb
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1: crop_sz = 10
    
    cx, cy = x + w / 2, y + h / 2
    x1 = round(cx - crop_sz / 2)
    y1 = round(cy - crop_sz / 2)
    
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz
    
    x1_pad = max(0, -x1); x2_pad = max(x2 - im.shape[1], 0)
    y1_pad = max(0, -y1); y2_pad = max(y2 - im.shape[0], 0)
    
    im_crop = im[max(0, y1):min(im.shape[0], y2), max(0, x1):min(im.shape[1], x2), :]
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    
    resize_factor = output_sz / crop_sz
    return cv2.resize(im_crop_padded, (output_sz, output_sz)), resize_factor

def process(img_arr):
    img_tensor = img_arr.astype(np.float32).transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_tensor = ((img_tensor / 255.0) - mean) / std
    return np.concatenate([img_tensor, img_tensor], axis=0) 

def transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz):
    box_extract_center = np.array(box_extract[0:2]) + 0.5*np.array(box_extract[2:4])
    box_in_center = np.array(box_in[0:2]) + 0.5*np.array(box_in[2:4])
    box_out_center = (crop_sz-1)/2 + (box_in_center - box_extract_center)*resize_factor
    box_out_wh = np.array(box_in[2:4])*resize_factor
    return np.concatenate((box_out_center - 0.5*box_out_wh, box_out_wh)) / (crop_sz-1)

def hann1d(sz, centered=True):
    if centered:
        return 0.5 * (1 - np.cos((2 * math.pi / (sz + 1)) * np.arange(1, sz + 1, dtype=np.float32)))
    return np.ones(sz, dtype=np.float32)

def hann2d(sz, centered=True):
    h1_0 = hann1d(sz[0], centered).reshape(1, 1, -1, 1)
    h1_1 = hann1d(sz[1], centered).reshape(1, 1, 1, -1)
    return h1_0 * h1_1

def cal_bbox(score_map_ctr, size_map, offset_map, feat_sz):
    idx = np.argmax(score_map_ctr.flatten())
    idx_y, idx_x = idx // feat_sz, idx % feat_sz
    size = size_map.reshape(2, -1)[:, idx]
    offset = offset_map.reshape(2, -1)[:, idx]
    return np.array([(idx_x+offset[0])/feat_sz, (idx_y+offset[1])/feat_sz, size[0], size[1]], dtype=np.float32), score_map_ctr.flatten()[idx]

def map_box_back(pred_box, state, resize_factor, search_size):
    cx_prev, cy_prev = state[0] + state[2] / 2, state[1] + state[3] / 2
    cx, cy, w, h = pred_box
    actual_search_size = search_size / resize_factor
    real_cx = cx_prev + (cx - 0.5) * actual_search_size
    real_cy = cy_prev + (cy - 0.5) * actual_search_size
    real_w = w * actual_search_size
    real_h = h * actual_search_size
    return [real_cx - real_w / 2, real_cy - real_h / 2, real_w, real_h]

# ===================== 强力锁定 & 居中算法 =====================
def get_centering_matrix(target_box, frame_w, frame_h, zoom_to_use):
    tx, ty, tw, th = target_box
    target_cx, target_cy = tx + tw / 2, ty + th / 2
    screen_cx, screen_cy = frame_w / 2, frame_h / 2

    # 1. 计算平移向量
    dx = screen_cx - target_cx
    dy = screen_cy - target_cy

    # 2. 构造变换矩阵
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    S = cv2.getRotationMatrix2D((screen_cx, screen_cy), 0, zoom_to_use)
    
    M_combined = np.vstack([S, [0, 0, 1]]) @ np.vstack([T, [0, 0, 1]])
    return M_combined[:2, :]

def calculate_required_zoom(target_box, frame_w, frame_h, padding_ratio=1.1):
    tx, ty, tw, th = target_box
    target_cx, target_cy = tx + tw / 2, ty + th / 2
    dx = abs(frame_w / 2 - target_cx)
    dy = abs(frame_h / 2 - target_cy)
    
    safe_w = (frame_w - 2 * dx)
    safe_h = (frame_h - 2 * dy)
    
    scale_x = frame_w / safe_w if safe_w > 1 else 10.0
    scale_y = frame_h / safe_h if safe_h > 1 else 10.0
    
    return max(scale_x, scale_y) * padding_ratio

# ===================== 交互与主逻辑 =====================
selection_done = False
confirm_done = False # 新增：用于确认锁定
temp_roi = None
roi_start = None
drawing = False # 记录是否正在拖拽

def mouse_drawing(event, x, y, flags, params):
    global temp_roi, selection_done, roi_start, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        drawing = True
        selection_done = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x1, y1 = roi_start
            temp_roi = (min(x1, x), min(y1, y), abs(x1 - x), abs(y1 - y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = roi_start
        temp_roi = (min(x1, x), min(y1, y), abs(x1 - x), abs(y1 - y))
        selection_done = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="input.mp4")
    parser.add_argument("--model", type=str, default="sutrack_b384.onnx")
    parser.add_argument("--padding", type=float, default=1.1)
    parser.add_argument("--alpha", type=float, default=0.1, help="Zoom平滑系数, 0~1, 越小越平滑")
    args = parser.parse_args()

    search_sz, temp_sz = (384, 192) if "384" in args.model else (224, 112)
    
    opts = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(args.model, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    if not ret: return
    H, W = frame.shape[:2]
    out = cv2.VideoWriter("final_locked_center.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))

    cv2.namedWindow("Locking...", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Locking...", mouse_drawing)
    
    # 修改后的交互逻辑：框选 -> 显示框 -> 按 Enter 确认
    while True:
        display_frame = frame.copy()
        if temp_roi is not None:
            tx, ty, tw, th = temp_roi
            cv2.rectangle(display_frame, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)
            if selection_done:
                cv2.putText(display_frame, "Press ENTER to Start or Re-draw", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Locking...", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and selection_done: # 13 是 Enter 键
            break
        elif key == ord('q'):
            return
    
    state = list(temp_roi)
    z_patch, z_scale = sample_target(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), state, 2.0, temp_sz)
    t_list = [process(z_patch)] * 2
    anno_list = [transform_image_to_crop(state, state, z_scale, temp_sz)] * 2
    hann = hann2d((search_sz//16, search_sz//16), True)

    # 初始化平滑 Zoom 值
    smooth_zoom = calculate_required_zoom(state, W, H, args.padding)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. 追踪
        rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_patch, x_scale = sample_target(rgb_raw, state, 4.0, search_sz)
        search_tensor = process(x_patch)

        ort_inputs = {
            "template": np.stack(t_list, 0)[None, ...],
            "search": search_tensor[None, None, ...],
            "template_anno": np.stack(anno_list, 0)[None, ...].astype(np.float32)
        }
        score, size, offset = session.run(None, ort_inputs)
        box, conf = cal_bbox(score * hann, size, offset, search_sz//16)
        
        # 2. 映射回原图坐标
        state = map_box_back(box, state, x_scale, search_sz)

        # 3. 平滑缩放逻辑 (EMA Filter)
        current_req_zoom = calculate_required_zoom(state, W, H, args.padding)
        smooth_zoom = args.alpha * current_req_zoom + (1 - args.alpha) * smooth_zoom
        
        M_center = get_centering_matrix(state, W, H, smooth_zoom)
        
        # 变换图像
        stable_display = cv2.warpAffine(frame, M_center, (W, H), borderMode=cv2.BORDER_REPLICATE)

        # 4. 验证渲染
        cv2.line(stable_display, (W//2-20, H//2), (W//2+20, H//2), (0, 0, 255), 1)
        cv2.line(stable_display, (W//2, H//2-20), (W//2, H//2+20), (0, 0, 255), 1)
        
        curr_w, curr_h = state[2] * smooth_zoom, state[3] * smooth_zoom
        cv2.rectangle(stable_display, 
                      (int(W/2 - curr_w/2), int(H/2 - curr_h/2)), 
                      (int(W/2 + curr_w/2), int(H/2 + curr_h/2)), (0, 255, 0), 1)

        out.write(stable_display)
        cv2.imshow("Locking...", stable_display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
