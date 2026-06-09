import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

os.makedirs("dataset", exist_ok=True)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze().numpy()

features_dict = {}
dataset_path = "dataset"

image_files = [f for f in os.listdir(dataset_path)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} images...")

for img_name in image_files:
    img_path = os.path.join(dataset_path, img_name)
    try:
        features_dict[img_name] = extract_features(img_path)
        print(f"✅ Processed: {img_name}")
    except Exception as e:
        print(f"❌ Error: {img_name}: {e}")

with open("features.pkl", "wb") as f:
    pickle.dump(features_dict, f)

print(f"\n✅ Done! {len(features_dict)} images processed!")
