# StyleID : A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition


**SIGGRAPH 2026 / ACM TOG Journal Track**

[![arXiv](https://img.shields.io/badge/arXiv-2604.21689-b31b1b.svg)](https://arxiv.org/abs/2604.21689)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://kwanyun.github.io/StyleID_page/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/kwanY/styleid)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/kwanY/stylebench-s)


StyleID is a CLIP-based image encoder designed to produce **identity embeddings that remain robust under stylization**.  
It can be used for **identity similarity measurement, retrieval, evaluation, and identity-aware conditioning** in generative pipelines.


<img src="https://cdn-uploads.huggingface.co/production/uploads/639d445524af4747d8d2af52/1pTEZ88YvwnbDPlV_UqpM.jpeg" width="800">

---

## Project Status

- [x] Quick start recognition model release 
- [x] ArXiv released
- [x] Dataset release
- [ ] Data generation code release
- [ ] Train code release
---

## Quick Start
```python
!pip install transformers==4.52.0
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("kwanY/styleid").to(device)
processor = CLIPProcessor.from_pretrained("kwanY/styleid")

img_path = "example.jpg"
img = Image.open(img_path).convert("RGB")

inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
```

#### Notes
- Not suitable for images with **multiple faces**
- Rough center crop near the face is recommended for better performance

--- 
## Intended Uses

StyleID embeddings can be used for:

- Identity similarity comparison
- Image retrieval
- Stylized identity evaluation
- Identity-aware conditioning for generative models
- Research on face recognition under domain shift and stylization

---

## License and Usage Notice

- StyleID is released for non-commercial research use.
- Do not use FFHQ-derived data for biometric human recognition


## Citation

If you find this work useful, please cite the paper:

```
@article{yun2026styleid,
  title={StyleID: A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition},
  author={Yun, Kwan and Lee, Changmin and Jeong, Ayeong and Kim, Youngseo and Lee, Seungmi and Noh, Junyong},
  journal={arXiv preprint arXiv:2604.21689},
  year={2026}
}
```

---
