# SUTrack-ONNX
This repository provides official ONNX versions of the SUTrack models. SUTrack is a powerful visual tracking framework. We have converted 5 different variants (Tiny, Base, Large) with different input resolutions to ONNX format for easier deployment across various platforms and inference engines (like TensorRT, OpenVINO, or ONNX Runtime).

## Model Zoo

The following table provides the ONNX model files, their configurations, and direct download links.

| Model File | Size | Search Size | Template Size | Download Link |
| :--- | :--- | :---: | :---: | :--- |
| **sutrack_t224.onnx** | 106 MB | 224 | 112 | [Download](https://github.com/whyb/SUTrack-ONNX/releases/download/onnx/sutrack_t224.onnx) |
| **sutrack_b224.onnx** | 340 MB | 224 | 112 | [Download](https://github.com/whyb/SUTrack-ONNX/releases/download/onnx/sutrack_b224.onnx) |
| **sutrack_b384.onnx** | 341 MB | 384 | 192 | [Download](https://github.com/whyb/SUTrack-ONNX/releases/download/onnx/sutrack_b384.onnx) |
| **sutrack_l224.onnx** | 1.18 GB | 224 | 112 | [Download](https://github.com/whyb/SUTrack-ONNX/releases/download/onnx/sutrack_l224.onnx) |
| **sutrack_l384.onnx** | 1.19 GB | 384 | 192 | [Download](https://github.com/whyb/SUTrack-ONNX/releases/download/onnx/sutrack_l384.onnx) |

> **Note:** **t** stands for Tiny, **b** for Base, and **l** for Large variants.

---

# SUTrack ONNX Video Inference Tool (video_track_onnx.py) User Guide

`video_track_onnx.py` is a real-time object tracking inference script based on ONNX Runtime. It supports the **SUTrack** series (Tiny, Base, Large) and automatically adapts the **Search Size** and **Template Size** based on the provided model filename.

## 1. Prerequisites

Ensure your Python (>=3.8) environment has the following core dependencies installed:

```bash
pip install onnxruntime opencv-python numpy argparse
```

## 2. Quick Start
Run the following command in your terminal to start tracking. Once the program launches, use your mouse to select the target in the first frame and press Enter to begin.
```bash
python video_track_onnx.py --model ./sutrack_l384.onnx --video ./data/test.mp4
```

## 3. Operation Workflow & Shortcuts
* Select Target (ROI): The program will pause on the first frame. Click and drag your left mouse button to draw a bounding box around the target.
* Confirm Selection: Press Enter or Space to confirm the ROI and start real-time tracking.
* Reset Selection: If you made a mistake, press C to cancel the current selection and re-draw.
* Exit Program: During tracking, press Q or ESC to close the window and exit.
   
## 3.操作流程与快捷键 (中文)
* 框选目标 (ROI)：程序启动后会暂停在第一帧。使用鼠标左键拖动框选目标。
* 确认选择：按下 回车 (Enter) 或 空格 (Space) 键确认，程序开始实时跟踪。
* 重新选择：如果框选错误，按下 C 键取消当前选择并重新框选。
* 退出程序：在跟踪过程中，按下 Q 键或 ESC 键可直接关闭窗口并退出。


## Origins & Weights

- **Original PyTorch Implementation:** [chenxin-dlut/SUTrack](https://github.com/chenxin-dlut/SUTrack)
- **Original Pre-trained Weights:** [Hugging Face Checkpoints](https://huggingface.co/xche32/SUTrack/tree/main/checkpoints/train/sutrack)

## Acknowledgements

We would like to express our gratitude to the authors of the SUTrack paper for their excellent work and for making their code and models available to the community:

**Xin Chen**, **Ben Kang**, **Wanting Geng**, **Jiawen Zhu**, **Yi Liu**, **Dong Wang**, and **Huchuan Lu**.

Special thanks to [Xin Chen](https://github.com/chenxin-dlut) for the original research and implementation.
