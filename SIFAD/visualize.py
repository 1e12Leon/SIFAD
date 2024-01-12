import re
import os
import matplotlib.pyplot as plt

xml_dir = 'D:\Deep_Learning_folds\Datasets\Dji_UAV_Dataset\Dji\VOCdevkit\VOC2007\Annotations'
# 示例数据
data = os.listdir(xml_dir)

# 提取高度数据
heights = []
pattern = r'h_([\d.]+)m'
for item in data:
    match = re.search(pattern, item)
    if match:
        height = float(match.group(1))
        heights.append(height)

# 统计区间
# heights.sort()
min_height = min(heights)
max_height = max(heights)
print(f"最小高度: {min_height}m")
print(f"最大高度: {max_height}m")

from sklearn.cluster import KMeans
import numpy as np
# 将高度数据转换为二维数组
heights_2d = np.array(heights).reshape(-1, 1)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=8, random_state=0).fit(heights_2d)
labels = kmeans.labels_

# 根据聚类结果将文件名保存在不同的列表中
clusters = {}
for i in range(len(labels)):
    if labels[i] in clusters:
        clusters[labels[i]].append(data[i])
    else:
        clusters[labels[i]] = [data[i]]

import xml.etree.ElementTree as ET
# 输出聚类结果
for key, value in clusters.items():
    for xml in clusters[key]:
        # print(xml)
        tree = ET.parse(os.path.join(xml_dir, xml))
        root = tree.getroot()
        scale_class = ET.Element('scale_class')
        scale_class.text = str(key)
        # print(root.text)
        # annotation = root.find('annotation')
        root.append(scale_class)
        tree.write(os.path.join(xml_dir, xml))


scale_classes = []
for xml in os.listdir(xml_dir):
    tree = ET.parse(os.path.join(xml_dir, xml))
    root = tree.getroot()
    annotation = root.find('scale_class')
    scale_classes.append(int(annotation.text))

cls = set(scale_classes)
print(cls)

"""# 可视化聚类结果
plt.scatter(range(len(heights)), heights, c=labels, cmap='viridis')
plt.title('Clustering Result')
plt.xlabel('Data Point')
plt.ylabel('Height')
plt.show()"""


