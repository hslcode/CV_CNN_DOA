# CV_CNN_DOA
A novel DOA estimation method based on deep complex-valued networks with sparse prior.
  --- 
## 🚀 Latest Update

### Power Normalization for Covariance Matrices @ 2025.03
To generalize to different hardware platforms and power levels, we strongly recommend power-normalizing the covariance matrix before model training and inference, using the following formula:

#### Mathematical Definition
Given covariance matrix $\hat{R} \in \mathbb{C}^{Batch \times M \times M}$ :  
```math
\hat{R}_{\text{normalized}} = \frac{\hat{R}}{\max\left(\text{diag}(\hat{R})\right) + \epsilon}
```
--- 
## Brief
- This is the code for IEEE ICICSP 2023 paper : "Robust DOA Estimation Using Deep ComplexValued Convolutional Networks with Sparse Prior". The link is： https://ieeexplore.ieee.org/document/10390873
- If this work is helpful to you, please star this  repositorie and cite our paper：

  @INPROCEEDINGS{10390873,
  author={Hu, Shulin and Zeng, Cao and Liu, Minti and Tao, Haihong and Zhao, Shihua and Liu, Yu},  
  booktitle={2023 6th International Conference on Information Communication and Signal Processing (ICICSP)},  
  title={Robust DOA Estimation Using Deep Complex-Valued Convolutional Networks with Sparse Prior},  
  year={2023},  
  volume={},  
  number={},  
  pages={234-239},  
  keywords={Training;Direction-of-arrival estimation;Quantization (signal);Simulation;Superresolution;Estimation;Feature extraction;direction of arrival estimation;deep complex-valued networks;sparse representation},  
  doi={10.1109/ICICSP59554.2023.10390873}}  


