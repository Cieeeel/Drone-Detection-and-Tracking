import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DroneDataset(Dataset):
    def __init__(self, root):
        self.imgs = list()
        self.boxes = list()
        self.labels = list()
        video_dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))]

        for video_path in video_dirs:
            # 加载frames和annotations
            with open(os.path.join(video_path, 'IR_label.json')) as f:
                annotations = json.load(f)

            for frame_file, exist, rect in zip(sorted(os.listdir(video_path)), annotations['exist'],
                                               annotations['gt_rect']):
                if frame_file.endswith('.jpg'):  # 替换为所需图片扩展名
                    self.imgs.append(os.path.join(video_path, frame_file))

                    if exist and rect[3] > 0:
                        self.labels.append(1)
                        self.boxes.append(rect)
                    else:
                        self.labels.append(0)
                        self.boxes.append([0, 0, 640, 512])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        frame = Image.open(img_path)
        frame = F.to_tensor(frame)

        x, y, w, h = self.boxes[idx]
        boxes = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)  # 转化为[x1, y1, x2, y2] 形式
        labels = torch.tensor(self.labels[idx], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        return frame, target

    def __len__(self):
        return len(self.imgs)


def build_data_loader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



# test_dataset = DroneDataset("data/test")
# test_data_loader = build_data_loader(test_dataset, batch_size=2, shuffle=False)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
from tqdm import tqdm
def train_model(model, data_loader, device, optimizer, num_epochs):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            targets = {k: v.to(device) for k, v in targets.items()}

            loss_dict = model(images, [targets])
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {losses.item()}")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2  # 1 class (drone) + background
model = get_model(num_classes)
model.load_state_dict(torch.load('model_weights.pth'))
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)   # 优化器使用SGD随机梯度下降方法

train_dataset = DroneDataset("data/train")
# train_data_loader = build_data_loader(train_dataset, batch_size=1, shuffle=True)
subset_size = 1000  # 选择1000张图进行训练

# 创建子集对象
subset = Subset(train_dataset, range(subset_size))
train_data_loader = build_data_loader(subset, batch_size=1, shuffle=True)

# ==========训练阶段============
# train_model(model, train_data_loader, device, optimizer, num_epochs=1)   # 模型训练

# torch.save(model.state_dict(), 'model_weights.pth')  # 保存训练权重


# 当dataset, data loader, model等均设置完毕

dataiter = iter(train_data_loader)  # 从data_loader创建迭代器
images, targets = next(dataiter)  # 获取首个batch的数据

# 将数据放入device
images = images.to(device)
targets = {k: v.to(device) for k, v in targets.items()}

# 确保模型处于evaluation mode
model.eval()

# 通过模型传递数据
with torch.no_grad():
    output = model(images)

# output是模型对第一批训练数据的预测，为dictionaries，包含boxes和labels
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# #获取批处理中的第一张图像
image = images[0].cpu().numpy().transpose((1, 2, 0))

# 获取批处理中的第一个预测
pred = output[0]

fig, ax = plt.subplots(1)
ax.imshow(image)

boxes = pred['boxes'].cpu().numpy()
labels = pred['labels'].cpu().numpy()

# 绘制检测边界框
for box, label in zip(boxes, labels):
    x, y, w, h = box
    rect = patches.Rectangle((x, y), w-x, h-y, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(x, y, f'{label}', color='white')

plt.show()

