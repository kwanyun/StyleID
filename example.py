import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("kwanY/styleid").to(device)
processor = CLIPProcessor.from_pretrained("kwanY/styleid")

img = Image.open("ex.png").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # optional but recommended

print(emb.shape)