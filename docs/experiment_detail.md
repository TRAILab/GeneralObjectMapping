# Additional Experiment Guidance

We employ optimization-based methods and offer several parameters to tailor the approach to real-world applications. Our methods are also sensitive to the initial states of poses and shapes. See below for recommendations on how to achieve better performance.


## Pose Initialization

We provide a simple ICP matcher to offer a coarse initial pose for objects. For more accurate pose estimation, more advanced methods are available, such as 3D object detectors.

## Hyperparameters

We provide parameters in the `json` files that can be adjusted to achieve better performance:

### Loss Weights

* `lr`: Learning rates for the geometric update step: smaller learning rates (LR) with longer iterations tend to yield more stable performance. Additionally, a smaller LR means the system will rely more on diffusion priors to generate complete and reasonable shapes, though these may differ from the observations.

* `geometric_constraints_w_3d`: Weight for the 3D point cloud loss.
* `geometric_constraints_w_2d`: Weight for the rendered RGB images loss.
* `geometric_constraints_w_depth`: Weight for the rendered depth images loss.

### Trade-off Between Accuracy and Efficiency

* `iter_num`: Number of iterations: Generally, longer iterations lead to better convergence, but this also increases computational time.

* `opt_num_per_diffuse`: The number of iterations for optimization per diffusion step: Larger values prioritize geometric constraints over priors.

* `geometric_constraints_3dloss_points_sample_each_iter`: Number of sampled points for the 3D loss.

* `geometric_constraints_2dloss_render_ray_num`: Number of sampled rays for the 2D rendering loss.


### Systems

* `flag_optimize_latent`: Whether to optimize the latent (shape).
* `flag_optimize_pose`: Whether to optimize the pose.
* `geometric_constraints_state`: Whether to use geometric constraints.
* `diffusion_prior_state`: Whether to use the diffusion prior.

## Output

The results are stored in the "saved" folder, including the following files:

* ```input_images.png```: Contains the input RGB, depth, masks, and cropped images.
* ```render_360deg.gif```: A rendered 360° view of the NeRF in normalized coordinates.
* ```render_mid.png```: A rendered NeRF with the pose applied to the N-th input image. This can be compared to the input RGB images.
* ```plot/loss_2d_3d.png```: The optimization loss with respect to iterations.
* ```plot/latent_update.png```: The gradients contributed by the geometric loss (Loss Grad) and the diffusion prior (Diffusion Prior) at each step.


## Further Questions

Feel free to contact the authors or raise any issues if you have further questions.
