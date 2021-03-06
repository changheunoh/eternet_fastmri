## ETER-net: An *E*nd-*T*o-*E*nd reconstruction network for MRI using *R*ecurrent neural network


### Abstract


A novel neural network architecture named 'ETER-net' is proposed as a unified solution to reconstruct an MR image directly from k-space data acquired with various k-space trajectories. <!--- [[link](https://github.com/changheunoh/eternet_fastmri/edit/master/README.md)] -->

The image reconstruction in MRI is performed by transforming the k-space data into image domain, where the domain transformation can be executed by Fourier transform when the k-space data satisfy the Shannon sampling theory. We propose an RNN-based architecture to achieve domain transformation and de-aliasing from undersampled k-space data in Cartesian and non-Cartesian coordinates. An additional CNN-based network and loss functions including adversarial, perceptual, and SSIM losses are proposed to refine and optimize the network performance.

This page is for validation of our method to a public dataset called 'FastMRI'. [[link](https://arxiv.org/abs/1811.08839)]



### Links

paper link : <!---  [arxiv_link](https://github.com/changheunoh/eternet_fastmri/edit/master/README.md)  -->

trained weight, input data, and label: [[link](https://drive.google.com/drive/folders/1jaKZ-J5sdypCoggGO8cIGWh3rsSemF0I?usp=sharing)]

### Network



![Image of network](model.png)


### Image



![Image of FastMRI](fastmri.png)





[edit](https://github.com/changheunoh/eternet_fastmri/edit/master/README.md)


