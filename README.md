# MACGAN-for-fault-diagnosis
Tensorflow Code for our paper *"*Multi-mode data augmentation and fault diagnosis of rotating machinery using modified ACGAN designed with new framework*"*.

Paper Link: (https://doi.org/10.1016/j.aei.2022.101552)

If you find our work useful in your research, please consider citing:

```
@inproceedings{MACGAN-for-fault-diagnosis-of-rotating-machinery,
  title={Multi-mode data augmentation and fault diagnosis of rotating machinery using modified ACGAN designed with new framework},
  author={Wei Li, Xiang Zhong, Haidong Shao, Baoping Cai, and Xingkai Yang},
  booktitle={Advanced Engineering Informatics},
  year={2022}
}
```

## Programming Environment

  * Python 3.8
  * Tensorflow 2.4.0
  * Numpy 1.19.5

## Datasets Preparation 

the bearing and gear fault datasets are collected from [Case Western Reserve University bearing data center](https://github.com/cathysiyu/Mechanical-datasets), [gear vibration dataset of University of Connecticut](https://figshare.com/articles/Gear_Fault_Data/6127874/1). 

| Health states of bearing | Health states of gear |
| ------------------------ | --------------------- |
| Normal                   | Chipping 1 (High)     |
| Inner race 0.007         | Chipping 3 (Middle)   |
| Inner race 0.0021        | Chipping 5 (Low)      |
| Outer race 0.007         | Crack                 |
| Outer race 0.014         | Missing               |
| Outer race 0.021         | Spalling              |
| Rolling element 0.014    | Healthy               |


In this paper, a signal-to-image conversion method [50] is used to transform the original 1D vibration signals into 2D gray images. code: signal-to-image conversion
[50] L. Wen, X. Li, L. Gao, Y. Zhang, A new convolutional neural network-based data-driven fault diagnosis method, IEEE Trans. Ind. Electron., 65, pp. 5990-5998, July. 2018. Paper Link: (https://ieeexplore.ieee.org/document/8114247)


## How to use

Before begin, please separate the training data of different health states into folders in folder`gray_images\\`. 
The code for training MACGAN and generate gray images is`training and generating.py`, you may also customize the parameters in config part in `training and generating.py`. The generated images will be saved in folder`generated images\\`.


## Acknowledgement

Our SpectralNormalization code is based on (https://github.com/IShengFang/SpectralNormalizationKeras)

## Contact

If you have any questions about the codes or would like to communicate about intelligent fault diagnosis, fault detection, please contact us: [liwei2020@hnu.edu.cn](mailto:liwei2020@hnu.edu.cn)
Mentor E-mail：[hdshao@hnu.edu.cn](mailto:hdshao@hnu.edu.cn)
