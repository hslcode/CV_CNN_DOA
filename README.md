# CV_CNN_DOA
A novel DOA estimation method based on deep complex-valued networks with sparse prior.
  --- 
## 🚀 Latest Update

### Power Normalization for Covariance Matrices @ 2025.03
To generalize to different hardware platforms and power levels, we strongly recommend power-normalizing the covariance matrix before model training and inference, using the following formula:

#### Mathematical Definition
Given sampling covariance matrix (SCM) $\hat{R} \in \mathbb{C}^{Batch \times M \times M}$ :  
```math
\hat{R}_{\text{normalized}} = \frac{\hat{R}}{\max\left(\text{diag}(\hat{R})\right) + \epsilon}
```
--- 
## Brief
- This is the code for IEEE ICICSP 2023 paper : "Robust DOA Estimation Using Deep ComplexValued Convolutional Networks with Sparse Prior". The link is： https://ieeexplore.ieee.org/document/10390873
- If this work is helpful to you, please star this  repositorie and cite our paper：

@inproceedings{hu2023robust,<br>
  title={Robust doa estimation using deep complex-valued convolutional networks with sparse prior},<br>
  author={Hu, Shulin and Zeng, Cao and Liu, Minti and Tao, Haihong and Zhao, Shihua and Liu, Yu},<br>
  booktitle={2023 6th International Conference on Information Communication and Signal Processing (ICICSP)},<br>
  pages={234--239},<br>
  year={2023},<br>
  organization={IEEE}<br>
}


