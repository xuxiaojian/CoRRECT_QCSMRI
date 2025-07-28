# [CoRRECT: A Deep Unfolding Framework for Motion-Corrected Quantitative R2* Mapping](https://link.springer.com/article/10.1007/s10851-025-01236-y)

- [Homepage](https://wustl-cig.github.io/correctwww/)
- ![Pipeline](https://github.com/xuxiaojian/2025-CoRRECT_QCSMRI/blob/main/pics/pipeline.png)


## Abstract
Quantitative MRI (qMRI) refers to a class of MRI methods for quantifying the spatial distribution of biological tissue parameters. Traditional qMRI methods usually deal separately with artifacts arising from accelerated data acquisition, involuntary physical motion, and magnetic field inhomogeneities, leading to sub-optimal end-to-end performance. This paper presents CoRRECT, a unified deep unfolding (DU) framework for qMRI consisting of a model-based end-to-end neural network, a method for motion artifact reduction, and a self-supervised learning scheme. The network is trained to produce R2* maps whose k-space data matches the real data by also accounting for motion and field inhomogeneities. When deployed, CoRRECT only uses the k-space data without any pre-computed parameters for motion or inhomogeneity correction. Our results on experimentally collected multi-gradient recalled echo (mGRE) MRI data show that CoRRECT recovers motion and inhomogeneity artifact-free R2* maps in highly accelerated acquisition settings. This work opens the door to DU methods that can integrate physical measurement models, biophysical signal models, and learned prior models for high-quality qMRI.

**Authored by**: Xiaojian Xu, Weijie Gan, Satya V. V. N. Kothapalli, Dmitriy A. Yablonskiy & Ulugbek S. Kamilov

## Instructions 
- This is the official repository for the implementation of the paper "CoRRECT: A Deep Unfolding Framework for Motion-Corrected Quantitative R2* Mapping." While we are currently unable to provide the dataset, the repository includes a detailed implementation along with the necessary configurations to reproduce our results.

- In the codebase, the method is referred to as QCSMRI, which stands for Quantitative and Compressed Sensing MRI.

- Please feel free to reach out with any questions or feedback.

### Citation
```
@article{xu2025correct,
  title={CoRRECT: A deep unfolding framework for motion-corrected quantitative R2* mapping},
  author={Xu, Xiaojian and Gan, Weijie and Kothapalli, Satya VVN and Yablonskiy, Dmitriy A and Kamilov, Ulugbek S},
  journal={Journal of Mathematical Imaging and Vision},
  volume={67},
  number={2},
  pages={20},
  year={2025},
  publisher={Springer}
}

```