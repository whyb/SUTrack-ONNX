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

## Origins & Weights

- **Original PyTorch Implementation:** [chenxin-dlut/SUTrack](https://github.com/chenxin-dlut/SUTrack)
- **Original Pre-trained Weights:** [Hugging Face Checkpoints](https://huggingface.co/xche32/SUTrack/tree/main/checkpoints/train/sutrack)

## Acknowledgements

We would like to express our gratitude to the authors of the SUTrack paper for their excellent work and for making their code and models available to the community:

**Xin Chen**, **Ben Kang**, **Wanting Geng**, **Jiawen Zhu**, **Yi Liu**, **Dong Wang**, and **Huchuan Lu**.

Special thanks to [Xin Chen](https://github.com/chenxin-dlut) for the original research and implementation.
