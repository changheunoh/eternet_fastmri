## ETER-net: An *E*nd-*T*o-*E*nd reconstruction network for MRI using *R*ecurrent neural network


### Abstract


In this work, a novel neural network architecture named 'ETER-net' is proposed as a unified solution to reconstruct an MR image directly from k-space data acquired with various k-space trajectories. The image reconstruction in MRI is performed by transforming the k-space data into image domain, where the domain transformation can be executed by Fourier transform when the k-space data satisfy the Shannon sampling theory. We propose an RNN-based architecture to achieve domain transformation and de-aliasing from undersampled k-space data in Cartesian and non-Cartesian coordinates. An additional CNN-based network and loss functions including adversarial, perceptual, and SSIM losses are proposed to refine and optimize the network performance.
We validated our method by applying it to a public dataset called 'FastMRI'. 

### Image

If you want to embed images, this is how you do it:

![Image of FastMRI](fastmri.png)


### Download link

paper link : arxiv [url](https://github.com/changheunoh/eternet_fastmri/edit/master/README.md)

trained weight, input data, and label: [link](https://github.com/changheunoh/eternet_fastmri/edit/master/README.md)

