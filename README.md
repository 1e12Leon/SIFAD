

<div align="center">


## Boost UAV-based Ojbect Detection via Scale-Invariant Feature Disentanglement and Adversarial Learning

[Fan Liu (刘凡)](https://multimodality.group/author/%E5%88%98%E5%87%A1/) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Liang Yao (姚亮)](https://multimodality.group/author/%E5%A7%9A%E4%BA%AE/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Chuanyi Zhang (张传一)](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Ting Wu (吴婷)](https://multimodality.group/author/%E5%90%B4%E5%A9%B7/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Xinlei Zhang (张新蕾)](https://multimodality.group/author/%E5%BC%A0%E6%96%B0%E8%95%BE/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Xiruo Jiang (姜希若)](http://www.milab-nust.com/web/graduatesshow.html?id=65) 
<img src="assets/NJUST.jpg" alt="Logo" width="15">, &nbsp; &nbsp; 
[Jun Zhou (周峻)](https://experts.griffith.edu.au/7205-jun-zhou) 
<img src="assets/griffith_logo.png" alt="Logo" width="15">

</div>

![image](https://github.com/user-attachments/assets/40772e6b-bcc7-452f-8972-ced9eac93318)

# News

- **2024/01/17**: We propose a Scale-Invariant Feature Disentanglement and Adversarial Learning method for UAV-OD. Codes and models will be open-sourced at this repository.


# Getting Started

## Install

- Clone this repo:

  ```bash
  git clone https://github.com/1e12Leon/SIFDAL.git
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n SIFDAL python=3.8 -y
  conda activate SIFDAL
  ```

- Install `CUDA Toolkit 11.3` ([link](https://developer.nvidia.com/cuda-11.3.0-download-archive)) and `cudnn==8.2.1` [(link)](https://developer.nvidia.com/rdp/cudnn-archive), then install `PyTorch==1.10.1`:

  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
  # if you prefer other cuda versions, please choose suitable pytorch versions
  # see: https://pytorch.org/get-started/locally/
  ```

## Data Preparation

This code repository uses VOC format object detection data.

```
VOCdevkit
    ├───VOC2007
    │   ├───ImageSets
    │   |     ├───Main
    │   |            test.txt
    │   |            train.txt
    │   |            trainval.txt
    │   |            val.txt
    │   ├───JPEGImages
    │   │       xxx1.jpg
    │   │       xxx2.jpg
    │   │       ...
    │   └───Annotations
                xxx1.xml
                xxx2.xml
                ...
```

# State-Air Dataset 

We constructed a multi-scene and multi-modal UAV-based object detection  dataset, State-Air. It was captured in a real-world outdoor setting with a wide variety of scenes and weather conditions. We are committed to further enhancing the scope and scale of State-Air, expanding both the coverage and depth of it.

## Download the Dataset 📂

*  [BaiduYun](https://pan.baidu.com/s/1yPEJJ1se5x7tsp2I8mQTsQ?pwd=j975)

## License 🚨
By downloading or using the Dataset, as a Licensee I/we understand, acknowledge, and hereby agree to all the terms of use. This dataset is provided "as is" and without any warranty of any kind, express or implied. The authors and their affiliated institutions are not responsible for any errors or omissions in the dataset, or for the results obtained from the use of the dataset. **The dataset is intended for academic research purposes only, and not for any commercial or other purposes.** The users of the dataset agree to acknowledge the source of the dataset and cite the relevant papers in any publications or presentations that use the dataset. The users of the dataset also agree to respect the intellectual property rights of the original data owners.

# Citation

```bibtex
@article{liu2024boost,
   title={Boost UAV-based Ojbect Detection via Scale-Invariant Feature Disentanglement and Adversarial Learning}, 
   author={Liu, Fan and Yao, Liang and Zhang, Chuanyi and Wu, Ting and Zhang, Xinlei and Jiang, Xiruo and Zhou, Jun},
   journal={arXiv preprint arXiv:2405.15465},
   year={2024} 
}
```

# Contact

Please Contact yaoliang@hhu.edu.cn
