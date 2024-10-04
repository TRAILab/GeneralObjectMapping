# General Object Mapping
Official Code Repository for the CoRL 2024 Paper: "Toward General Object-Level Mapping from Sparse Views with 3D Diffusion Priors".

We are currently organizing our code and model and will release them soon. Thank you for your patience!


## Abstract

Object-level mapping builds a 3D map of objects in a scene with detailed shapes and poses from multi-view sensor observations. Conventional methods struggle to build complete shapes and estimate accurate poses due to partial occlusions and sensor noise. They require dense observations to cover all objects, which is challenging to achieve in robotics trajectories.  Recent work introduces generative shape priors for object-level mapping from sparse views, but is limited to single-category objects. In this work, we propose a General Object-level Mapping system, GOM, which leverages a 3D diffusion model as shape prior with multi-category support and outputs Neural Radiance Fields (NeRFs) for both texture and geometry for all objects in a scene. 
GOM includes an effective formulation to guide a pre-trained diffusion model with extra nonlinear constraints from sensor measurements without finetuning. We also develop a probabilistic optimization formulation to fuse multi-view sensor observations and diffusion priors for joint 3D object pose and shape estimation. 
GOM demonstrates superior multi-category mapping performance from sparse views, and achieves more accurate mapping results compared to state-of-the-art methods on the real-world benchmarks. 

[PDF](https://openreview.net/forum?id=rEteJcq61j) [Appendix](https://openreview.net/forum?id=rEteJcq61j) 


## Reference

Liao, Ziwei, Binbin Xu, and Steven L. Waslander. "Toward General Object-level Mapping from Sparse Views with 3D Diffusion Priors." 8th Annual Conference on Robot Learning.
