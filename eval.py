import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
os.makedirs('results', exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def save_predictions_to_file(video_dir, model, device):
    model.eval()  # 模型设置为evaluation mode

    video_name = os.path.basename(video_dir)

    # 载入frames
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])

    results = []

    for frame_file in frame_files:
        frame_path = os.path.join(video_dir, frame_file)
        frame = Image.open(frame_path).convert("RGB")
        frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)  # 添加batch dimension并送入设备

        with torch.no_grad():  # 不计算梯度
            prediction = model(frame_tensor)

        scores = prediction[0]['scores'].cpu().numpy()
        boxes = prediction[0]['boxes'].cpu().numpy()

        if len(scores) > 0:
            # 只保留得分最高的box
            max_score_idx = scores.argmax()
            max_score_box = boxes[max_score_idx]

            # 将检测框从(xmin, ymin, xmax, ymax)转化为 (x, y, width, height)
            xmin, ymin, xmax, ymax = max_score_box
            box = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]

            results.append([box])  # 在框周围添加一个列表（因为输出格式需要一个框列表）
        else:
            results.append([])  # 这一帧没有检测到

    # 将结果写入文件
    with open(os.path.join('results', f'{video_name}.txt'), 'w') as f:

        json.dump({"res": results}, f)


# ==========Test阶段===========
root = "E:/tracking/FRCNN/Data/test/"    # test数据集目录
video_dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2  # 1 class (drone) + background
model = get_model(num_classes)
model.load_state_dict(torch.load('model_weights.pth'), strict=False)
model.to(device)


for video_dir in video_dirs:
    save_predictions_to_file(video_dir, model, device)
