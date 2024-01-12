import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from nets.yolo import YoloBody

input_shape = [640, 640]
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 8
phi = 'l'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YoloBody(anchors_mask, num_classes, phi, False).to(device)

# 图像路径
image_path = r'imgs/state-air-test.jpg'
 # 'VOCdevkit/VOC2007/JPEGImages/frame_20190905143505_x_0003197.jpg'

# 加载图像
image = Image.open(image_path)
iw, ih  = image.size
w, h    = input_shape
new_image = image.resize((w, h), Image.BICUBIC)

# 将图像转换为numpy数组
image_array = np.array(new_image)
# 将numpy数组转换为PyTorch张量
input_tensor = torch.from_numpy(image_array)
# 获取模型权重的数据类型
weight_dtype = next(model.parameters()).dtype
# 添加batch维度
input_tensor = torch.unsqueeze(input_tensor, dim=0).to(dtype=weight_dtype, device=device)
input_tensor = input_tensor.permute(0, 3, 1, 2)
# 打印输入张量的形状
print(input_tensor.shape)

out, rcho, scale_logits, Invariant_feature = model(input_tensor)
print(Invariant_feature)
# 假设特征层的张量名为feature_map，shape为(batch_size, num_channels, height, width)
feature_map = Invariant_feature.cpu()

# 将特征层的张量转换为NumPy数组
feature_map_np = feature_map.detach().numpy()

# 叠加通道并显示
combined_feature_map = np.sum(feature_map_np[0], axis=0)  # 叠加第一个样本的所有通道
plt.imshow(combined_feature_map)
plt.show()


"""# 假设特征层的张量名为feature_map，shape为(batch_size, num_channels, height, width)
feature_map2 = Scale_feature.cpu()

# 将特征层的张量转换为NumPy数组
feature_map_np2 = feature_map2.detach().numpy()

# 叠加通道并显示
combined_feature_map2 = np.sum(feature_map_np2[0], axis=0)  # 叠加第一个样本的所有通道
plt.imshow(combined_feature_map2)
plt.show()"""