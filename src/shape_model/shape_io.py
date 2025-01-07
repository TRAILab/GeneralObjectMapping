"""

Model connects to Shap-E.

Basic function: return a shape from an input RGB image.

"""

# for cache
import hashlib
import os
import time

import cv2
import numpy as np
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config

# for denoising
# shap-e/shap_e/diffusion/k_diffusion.py
from shap_e.diffusion.k_diffusion import to_d
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_config, load_model
from shap_e.models.transmitter.base import Transmitter
from shap_e.util.collections import AttrDict
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)

from src.optimizer.camera import create_cameras_with_grad_from_pose
from src.optimizer.renderer import render_views_from_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Shap_E:
    def __init__(self, model_path="./", fix_param=True, grid_size=None, model="image"):
        """

        @fix_param: by default, all parameters are fixed and do not calculate gradients.
        """
        self.model_path = model_path

        """
        Load all parameters of the model.
        """
        print("loading parameters of Shap-E model ...")

        self.xm = load_model("transmitter", device=device)

        if model == "image":
            self.model = load_model("image300M", device=device)
        elif model == "text":
            # if open text option, load this one
            print("Load text model ...")
            self.model = load_model("text300M", device=device)

        self.diffusion = diffusion_from_config(load_config("diffusion"))

        # for diffusion prior
        # shap-e/shap_e/diffusion/k_diffusion.py
        from shap_e.diffusion.k_diffusion import GaussianToKarrasDenoiser

        self.model_denoiser = GaussianToKarrasDenoiser(self.model, self.diffusion)

        if fix_param:
            # fix all parameters
            for param in self.xm.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = False
            # for param in self.diffusion.parameters():
            #     param.requires_grad = False

        """
        Further config for computation
        """
        if grid_size is not None:
            # Grid size is used for mesh generation; Sample around the whole space with default 128*128*128
            print(" * Change grid size:", grid_size)
            self.xm.renderer.grid_size = grid_size

        """
        Init cache variable
        """
        self.cache_image_embeddings = {}  # image_binary_id -> feature; used in diffusion prior

        # print('loading done.')

        # Cache mask rays
        self.cache_mask_rays = {}

    # Update: get latent from text
    def get_latent_from_text(
        self, text, batch_size=1, guidance_scale=3.0, cache=False, cache_dir="./output/cache/"
    ):

        if cache:
            # check local chache dir to see if we have done for the input image
            os.makedirs(cache_dir, exist_ok=True)

            # get an unique text identifier
            # image_id = hashlib.md5(image.tobytes()).hexdigest()
            text_id = text.replace(" ", "_")

            save_pt_cache = os.path.join(cache_dir, text_id, "latent.pt")

            if os.path.exists(save_pt_cache):
                print("load cache from:", save_pt_cache)
                latent = torch.load(save_pt_cache)
                return latent

        # (Batchsize, 1024x1024)
        latents = sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(
                texts=[text] * batch_size, clip_embedding_offset=None
            ),  # Update: Support CLIP feature offset
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        latent = latents[0]

        if cache:
            # create dir for parent
            os.makedirs(os.path.dirname(save_pt_cache), exist_ok=True)

            # save it
            torch.save(latent, save_pt_cache)

            print("save cache to:", save_pt_cache)

        return latent

    def get_latent_from_image(
        self,
        rgb_image,
        batch_size=1,
        guidance_scale=3.0,
        cache=False,
        bgr_to_rgb=False,
        clip_embedding_offset=None,
        cache_dir="./output/cache/",
    ):
        """
        Input:
            rgb_image : (H,W,3)

            bgr_to_rgb: Note if the image is loaded by opencv, the channels need to be changed to RGB.

        Output:
            latent : (1024x1024,)
        """

        image = rgb_image

        if bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if cache:
            # check local chache dir to see if we have done for the input image

            temp_cache_dir = cache_dir
            os.makedirs(temp_cache_dir, exist_ok=True)

            # get an unique input image identifier
            image_id = hashlib.md5(image.tobytes()).hexdigest()

            save_pt_cache = os.path.join(temp_cache_dir, image_id, "latent.pt")

            if os.path.exists(save_pt_cache):
                print("load cache from:", save_pt_cache)
                # directly load and output
                latent = torch.load(save_pt_cache)
                return latent

        # (Batchsize, 1024x1024)
        latents = sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(
                images=[image] * batch_size, clip_embedding_offset=clip_embedding_offset
            ),  # Update: Support CLIP feature offset
            progress=False,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        latent = latents[0]

        if cache:
            # create dir for parent
            os.makedirs(os.path.dirname(save_pt_cache), exist_ok=True)

            # save it
            torch.save(latent, save_pt_cache)

            # note we need to change back to bgr
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # save input image
            save_image_cache = os.path.join(temp_cache_dir, image_id, "image.png")
            cv2.imwrite(save_image_cache, image_bgr)

            print("save cache to:", save_pt_cache)

        return latent

    def get_shape_from_latent(self, latent):
        """
        return: tri mesh.
        """

        t = decode_latent_mesh(self.xm, latent).tri_mesh()

        return t

    def save_shape(self, tri_mesh, file_name):
        """
        filename: xxx.ply
        """
        with open(file_name, "wb") as f:
            tri_mesh.write_ply(f)
        # with open(f'example_mesh_{i}.obj', 'w') as f:
        #     t.write_obj(f)

        return

    """
    Add loss functions to support optimization

    Update: Support RGB in Pointcloud
    """

    def get_3d_surface_loss(self, latent, pts_3d, loss_type="mse"):
        """
        Input:
            latent : (1024x1024,)
            pts_3d : (N,3) or (N,6) with RGB
            loss_type : 'mse' or 'huber'
        Output:
            loss : (1,)
        """

        from functools import partial

        import numpy as np
        from shap_e.models.nn.meta import subdict
        from shap_e.models.query import Query

        renderer = self.xm.renderer
        options = AttrDict(rendering_mode="stf", render_with_direction=False)
        params = (
            self.xm.encoder if isinstance(self.xm, Transmitter) else self.xm
        ).bottleneck_to_params(latent[None])

        # if rendering_mode == "stf":
        sdf_fn = tf_fn = nerstf_fn = None
        if renderer.nerstf is not None:
            nerstf_fn = partial(
                renderer.nerstf.forward_batched,
                params=subdict(params, "nerstf"),
                options=options,
            )
        else:
            sdf_fn = partial(
                renderer.sdf.forward_batched,
                params=subdict(params, "sdf"),
                options=options,
            )
            tf_fn = partial(
                renderer.tf.forward_batched,
                params=subdict(params, "tf"),
                options=options,
            )

        # query_batch_size = batch.get("query_batch_size", batch.get("ray_batch_size", 4096))
        query_batch_size = 4096
        batch_size = 1

        # from shap-e/shap_e/models/stf/renderer.py import
        from shap_e.models.stf.renderer import volume_query_points

        query_points = pts_3d.to(latent.device)

        fn = nerstf_fn if sdf_fn is None else sdf_fn
        sdf_out = fn(
            query=Query(position=query_points[None].repeat(batch_size, 1, 1)),
            query_batch_size=query_batch_size,
            options=options,
        )
        raw_signed_distance = sdf_out.signed_distance

        # loss
        # loss_type = 'l2'
        if loss_type == "mse":
            # l2 loss: all sdf values should equal to zero
            loss = torch.mean(raw_signed_distance**2)
        elif loss_type == "huber":
            loss_func = torch.nn.HuberLoss(reduction="mean")
            loss = loss_func(raw_signed_distance, torch.zeros_like(raw_signed_distance))
        else:
            raise NotImplementedError

        return loss, raw_signed_distance

    def render_rays(self, latent, rays, ray_batch_size: int = 1000, cam_z: torch.Tensor = None):
        """
        Unit function used for 2D loss;
        The rays are generated from masked images, and sample pixels.

        Input:
            @ latent : (1024x1024,)
            @ rays : (1, N, 2, 3)


        Output:
            @ rays_info : (N, 4)
        """

        params = (
            self.xm.encoder if isinstance(self.xm, Transmitter) else self.xm
        ).bottleneck_to_params(latent[None])

        rendering_mode = "nerf"
        options = AttrDict(rendering_mode=rendering_mode, render_with_direction=False)

        output = render_views_from_rays(
            rays,
            self.xm.renderer.render_rays,
            ray_batch_size,
            params=params,
            options=options,
            cam_z=cam_z,
        )

        return output

    def generate_rays_and_cache(self, mask, depth, bbox_scale):
        """
        Generate the indices for the valid areas inside the images.
        """

        """Check if we have seen this mask """
        mask_id = hashlib.md5(mask.tobytes()).hexdigest()

        if mask_id in self.cache_mask_rays:
            # print("Use cache for mask rays:", mask_id)
            return self.cache_mask_rays[mask_id]

        # Convert inputs to torch tensors
        mask = torch.from_numpy(mask)

        """
        Update: Not all depths values inside mask is valid; Ignore those depths with 0
        """
        b_consider_valid_depth = False
        if b_consider_valid_depth:
            mask_depth = depth > 0
            mask = mask & mask_depth

        # Find the bounding box of the mask
        y_indices, x_indices = torch.where(mask)
        bbox = [
            torch.min(x_indices),
            torch.min(y_indices),
            torch.max(x_indices),
            torch.max(y_indices),
        ]

        # Scale the bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        bbox = [
            bbox[0] - bbox_scale * width / 2,
            bbox[1] - bbox_scale * height / 2,
            bbox[2] + bbox_scale * width / 2,
            bbox[3] + bbox_scale * height / 2,
        ]

        # Clip the bounding box to the dimensions of the mask
        bbox = [
            torch.clamp(bbox[0], 0, mask.shape[1]),
            torch.clamp(bbox[1], 0, mask.shape[0]),
            torch.clamp(bbox[2], 0, mask.shape[1]),
            torch.clamp(bbox[3], 0, mask.shape[0]),
        ]

        foreground_indices = torch.where(mask)  # y,x

        # Create x_indices and y_indices; no need to transpose for this
        x_indices, y_indices = torch.meshgrid(
            torch.arange(mask.shape[1]), torch.arange(mask.shape[0])
        )
        background_indices = torch.where(
            (x_indices >= bbox[0])
            & (x_indices < bbox[2])
            & (y_indices >= bbox[1])
            & (y_indices < bbox[3])
            & (~mask).t()
        )

        # Cache the results
        self.cache_mask_rays[mask_id] = foreground_indices, background_indices

        return foreground_indices, background_indices

    def generate_rays(
        self,
        mask,
        rgb,
        depth,
        t_cam_obj,
        K,
        ray_num=1000,
        bbox_scale=1.2,
        dtype=torch.float32,
        device=torch.device("cuda"),
        dense=False,
    ):
        """
        1. pixel sample: Get a bbox from the mask; scale it with bbox_scale; randomly sample ray_num points inside the bbox.

        2. ray generation: use t_cam_obj, K to generate rays (origin, direction) in the object frame.

        @ dense: if True, sample pixels for all the points inside mask
        """

        foreground_indices, background_indices = self.generate_rays_and_cache(
            mask, depth, bbox_scale
        )

        # t_cam_obj = torch.from_numpy(t_cam_obj)
        K = torch.from_numpy(K).to(dtype=dtype, device=device)

        """Sample indices, and then get points"""
        if dense:
            """
            Original Dense: Cover all the mask; But the calculation is different and more complicated
            """

            foreground_ray_num = len(foreground_indices[0])
            background_ray_num = len(background_indices[0])

            # sample_method = 'equal_skip_16'

            # Update: Equally sample foreground and background for evaluation metrics,
            # But make sure the sampled number is 7:3, and with a maximum value.
            sample_method = "equal_skip_dynamic"
            dense_ray_num = 10000
            foreground_ray_num = int(dense_ray_num * 0.7)
            background_ray_num = dense_ray_num - foreground_ray_num

        else:

            # Calculate the number of rays for the foreground and background
            foreground_ray_num = int(ray_num * 0.7)
            background_ray_num = ray_num - foreground_ray_num

            sample_method = "random"

        # Sample points for the foreground (inside mask)
        if sample_method == "random":
            foreground_sampled_indices = torch.randint(
                0, len(foreground_indices[0]), (foreground_ray_num,)
            )
        elif sample_method == "equal_skip_16":
            foreground_sampled_indices = torch.arange(0, len(foreground_indices[0]), 16)
        elif sample_method == "equal_skip_dynamic":
            skip = int(len(foreground_indices[0]) / foreground_ray_num) + 1
            foreground_sampled_indices = torch.arange(0, len(foreground_indices[0]), skip)

        foreground_sampled_points = torch.stack(
            [
                foreground_indices[0][foreground_sampled_indices],
                foreground_indices[1][foreground_sampled_indices],
            ],
            dim=-1,
        )
        # transpose to change y,x to x,y
        foreground_sampled_points = foreground_sampled_points.flip(1)

        # Sample points for the background (outside mask, but inside bbox)
        if sample_method == "random":
            background_sampled_indices = torch.randint(
                0, len(background_indices[0]), (background_ray_num,)
            )
        elif sample_method == "equal_skip_16":
            background_sampled_indices = torch.arange(0, len(background_indices[0]), 16)
        elif sample_method == "equal_skip_dynamic":
            skip = int(len(background_indices[0]) / background_ray_num) + 1
            background_sampled_indices = torch.arange(0, len(background_indices[0]), skip)

        background_sampled_points = torch.stack(
            [
                background_indices[0][background_sampled_indices],
                background_indices[1][background_sampled_indices],
            ],
            dim=-1,
        )

        # Combine the foreground and background points
        sampled_points = torch.cat([foreground_sampled_points, background_sampled_points], dim=0)

        ############################################################

        # Calculate Rays According to Points

        # real ray_num
        ray_num_real = sampled_points.shape[0]

        # Convert the points to homogeneous coordinates
        points_h = torch.cat([sampled_points, torch.ones(ray_num_real, 1)], dim=-1).to(
            dtype=dtype, device=device
        )

        # Compute the direction of the rays in camera coordinates
        direction_cam = torch.inverse(K) @ points_h.t()
        direction_cam = direction_cam.to(device)

        # Compute the origin and direction of the rays in object coordinates
        origin_obj = -torch.inverse(t_cam_obj[:3, :3]) @ t_cam_obj[:3, 3]
        direction_obj = torch.inverse(t_cam_obj[:3, :3]) @ direction_cam

        # Transpose the results to get rays of shape [ray_num, 3]
        origin_obj = origin_obj.t()
        # repeat the origin to match the shape of direction
        origin_obj = origin_obj.repeat(ray_num_real, 1)

        direction_obj = direction_obj.t()

        # Stack the origins and directions together to get rays of shape [ray_num, 2, 3]
        rays = torch.stack([origin_obj, direction_obj], dim=1)

        """
        Final Step: We get GT Obs for those rays.

        Generate gt rgb values for sampled points
        """
        run = (rgb is not None) or (depth is not None)

        if run:
            # Separate the foreground and background points
            foreground_points = sampled_points[:foreground_ray_num]
            background_points = sampled_points[foreground_ray_num:]

        # For the foreground points, choose observed_rgb from rgb_image
        if rgb is not None:
            foreground_rgb = rgb[foreground_points[:, 1], foreground_points[:, 0], :]  # 0-255!
            # Convert to torch tensor and move to the same device as latent
            foreground_rgb = torch.from_numpy(foreground_rgb).to(
                dtype=t_cam_obj.dtype, device=t_cam_obj.device
            )

            # For the background points, create a tensor of zeros with the same shape
            background_rgb = torch.zeros(
                (background_points.shape[0], 3), dtype=t_cam_obj.dtype, device=t_cam_obj.device
            )

            # Concatenate the foreground and background RGB values together
            obs_rgb = torch.cat([foreground_rgb, background_rgb], dim=0)
        else:
            obs_rgb = None

        if depth is not None:
            # For the foreground points, choose observed_depth from depth_image
            foreground_depth = depth[foreground_points[:, 1], foreground_points[:, 0]]
            foreground_depth = torch.from_numpy(foreground_depth).to(
                dtype=t_cam_obj.dtype, device=t_cam_obj.device
            )

            # For the background points, create a tensor of zeros with the same shape
            background_depth = torch.zeros(
                (background_points.shape[0],), dtype=t_cam_obj.dtype, device=t_cam_obj.device
            )

            # Concatenate the foreground and background depth values together
            obs_depth = torch.cat([foreground_depth, background_depth], dim=0)
        else:
            obs_depth = None

        return rays, obs_rgb, obs_depth

    @torch.no_grad()
    def _check_2d_rays_render(self, latent, t_cam_obj, K, mask):
        """
        Manually generate all pixels in the mask, and render a image to check if it's correct
        """
        # generate rays from pose, box, sample ...
        rgb = None
        rays, obs_rgb = self.generate_rays(mask, rgb, t_cam_obj, K, dense=True, bbox_scale=1.0)

        # input rays: rays in the object frame, can be directly used by NeRF network
        rays_info = self.render_rays(latent, rays)

        # construct an image based on the result!
        arr = rays_info.channels.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        bbox = [797.5, 499, 1296, 968]  # x1,y1,x2,y2

        # Calculate the dimensions of the bounding box
        bbox_width = int(bbox[2] - bbox[0])
        bbox_height = int(bbox[3] - bbox[1])

        # Reshape the array to the dimensions of the bounding box
        arr_im_bbox = arr.reshape(bbox_width + 1, bbox_height, 3)

        # rgb to bgr for opencv to save
        # save to image
        import cv2

        arr_im_bbox = cv2.cvtColor(arr_im_bbox, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./output/rays_render.png", arr_im_bbox)

        return

    def get_2d_render_loss(
        self,
        latent,
        t_cam_obj: torch.Tensor,
        K: torch.Tensor,
        mask,
        rgb,
        depth,
        ray_num=1000,
        dense=False,
        open_psnr=False,
        loss_type="mse",
    ):
        """
        Efficient version:
            1, get a bounding box from the mask; and scale it larger a little as ROI
            2, randomly sample pixels inside the ROI
            3, generate 3D rays
            4, render rays information
            5, construct loss

        Input:
            @ mask: (H,W) 0-1 mask
            @ loss_type: 'mse' or 'huber'

        Output:
            @ loss_rgb, loss_depth

        """
        # step 1: render an image
        # im_render = self.render_image(latent, t_cam_obj, K)

        time_start = time.time()

        # generate rays from pose, box, sample ...; sampled_points: x and y
        rays, obs_rgb, obs_depth = self.generate_rays(
            mask, rgb, depth, t_cam_obj, K, ray_num=ray_num, dense=dense
        )

        time_generate_rays = time.time()

        # input rays: rays in the object frame, can be directly used by NeRF network
        rays_info = self.render_rays(latent, rays)

        time_render_rays = time.time()

        """Calculate Losses"""

        # a loss function for two rgb images
        if loss_type == "mse":
            loss_image = torch.nn.MSELoss(reduction="mean")
        elif loss_type == "huber":
            loss_image = torch.nn.HuberLoss(reduction="mean")

        # step 2: calculate loss between two images
        rgb_rendered = rays_info.channels.squeeze()
        if rays_info.distances is not None:  # 1,1,1,N,3
            depth_rendered = rays_info.distances.squeeze()  # 1,1,1,N,1
            loss_depth = loss_image(obs_depth, depth_rendered)
        else:
            loss_depth = 0

        loss_rgb = loss_image(obs_rgb, rgb_rendered)  # MSE

        # loss_rgb = torch.mean((obs_rgb - rgb_rendered)**2)

        # loss = loss_image(im_render, rgb_image)
        # loss = loss_rgb + loss_depth

        time_end = time.time()

        # print(" >> Time for generating rays:", time_generate_rays - time_start)
        # print(" >> Time for rendering rays:", time_render_rays - time_generate_rays)
        # print(" >> Time for calculating loss:", time_end - time_render_rays)

        # TODO: if specified, further calculate PSNR metric
        if open_psnr:
            # calculate psnr
            mse = loss_rgb
            psnr = 20 * torch.log10(255 / torch.sqrt(mse))
            return loss_rgb, loss_depth, psnr
        else:
            return loss_rgb, loss_depth

    def render_image(
        self,
        latent,
        t_cam_obj: torch.Tensor,
        K: torch.Tensor,
        resize_scale=1.0,
        background: torch.Tensor = None,
    ):
        """
        Input:
            latent : (1024x1024,)
            t_cam_obj : (4,4)

            background: default black; you can specify Tensor([255,255,255]) as white
        Output:
            im_render : (H,W,3)
        """
        from shap_e.util.notebooks import (
            decode_latent_images,
            decode_latent_images_with_grad,
        )

        render_mode = "nerf"  # you can change this to 'stf' for mesh rendering

        """
        Note for a 11G gpu, support 64x64 w/o gradients;
        """
        # size = 128 # this is the size of the renders; higher values take longer to render.
        # ray_batch_size = 1295  # default: 4096
        ray_batch_size = 4096

        # Test gradients propagation

        cameras = create_cameras_with_grad_from_pose(t_cam_obj, K, resize=resize_scale)

        # debug with no gradients
        with torch.no_grad():
            images_with_grad = decode_latent_images_with_grad(
                self.xm,
                latent,
                cameras,
                rendering_mode=render_mode,
                ray_batch_size=ray_batch_size,
                background=background,
            )

        return images_with_grad

    def render_images_for_vis(self, latent, size=64, output_file="./render.gif", background=None):
        """
        @ size: resolution of renderd images
        """
        render_mode = "nerf"  # you can change this to 'stf' for mesh rendering
        # size = 64 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, device)
        images = decode_latent_images(
            self.xm, latent, cameras, rendering_mode=render_mode, background=background
        )

        images[0].save(
            output_file, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
        )

        return

    @torch.no_grad()
    def get_diffusion_prior_grad(
        self, latent, cond_data, sigma=1.0, grad_method="start", next_sigma=None
    ):
        """
        Run one step diffusion to get the predicted noise from the output.
        Then calculate the gradient of the latent w.r.t. the predicted noise.

        Args:
            - latent : (1024x1024,)
            - cond_data : an image (H,W,3), or a text description

            - grad_method : 'start', 'step', 'noise_plus_denoise', 'euler'
            - next_sigma: used for euler method.
        Output:
            - grad : (1024x1024,), the meanings are different for different grad_method
        """

        x_t = latent.unsqueeze(0)

        sigma = torch.Tensor([sigma]).to(device)

        # Cache for computational efficiency
        if isinstance(cond_data, np.ndarray):
            # Case 1: Input conditional data is an image
            data_id = hashlib.md5(cond_data.tobytes()).hexdigest()
        else:
            # Case 2: Input conditional data is a text description
            data_id = cond_data

        if data_id in self.cache_image_embeddings:
            # directly use the cache
            model_kwargs = self.cache_image_embeddings[data_id]
        else:
            # calculate new feature

            # clear the old cache
            self.cache_image_embeddings = {}

            batch_size = 1

            if isinstance(cond_data, np.ndarray):
                # Case 1: Input conditional data is an image
                cond_image = cond_data
                model_kwargs = dict(images=[cond_image] * batch_size)

                if hasattr(self.model, "cached_model_kwargs"):
                    model_kwargs = self.model.cached_model_kwargs(batch_size, model_kwargs)

            elif isinstance(cond_data, str):
                # Case 2: Input conditional data is a text description
                cond_text = cond_data

                # Option 1: directly use clip embed_text
                text_embeddings = self.model.wrapped.clip.embed_text([cond_text])  # (1,768)

                # consider batch
                # text_embeddings = text_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

                model_kwargs = dict(embeddings=text_embeddings)

            self.cache_image_embeddings[data_id] = model_kwargs

        # Run through the Diffusion model, and get pred_xstart and mean(x_previous)
        denoised_xt_pre, denoised_xstart = self.model_denoiser.denoise_pri(
            x_t, sigma, clip_denoised=True, model_kwargs=model_kwargs
        )

        # Calculate gradients from the diffusion outputs, according to different methods
        if grad_method == "start":
            denoised_xt = denoised_xstart
            grad = denoised_xt - x_t

        elif grad_method == "step":
            denoised_xt = denoised_xt_pre
            grad = denoised_xt - x_t

        elif grad_method == "noise_plus_denoise":
            # output predicted noise
            # predicted_noise = x_t - denoised_xt_pre
            x_0_curve = denoised_xstart

            grad = x_0_curve

        elif grad_method == "euler":
            """
            Euler method:

            Calculate an estimated x_0, and normalize with the sigma, to get a grad direction.
            Then multiply the direction by a sigma diff (from T to T-1), as a final grad with scale.

            **Ouput: a grad with scale, so that x_T-1 = x_T + grad**

            Heun's method:

            calculate an estimated x_t-1, and use it to denoise again, then calculate
            an average gradient.
            """
            denoised = denoised_xstart

            sigma_hat = sigma

            d = to_d(x_t, sigma_hat, denoised)

            dt = next_sigma - sigma_hat

            # Euler method
            # x = x + d * dt
            grad = d * dt

        # normalize
        # normalize=False
        # if normalize:
        #     grad = grad / torch.norm(grad)

        grad = grad.squeeze()

        return grad

    def update_latent_with_diffusion_prior(
        self, latent, diffusion_prior_grad, grad_method="start", **kwargs
    ):
        """
        After using `get_diffusion_prior_grad` to get the outputs from the diffusion model, use this function to
        update the latent with the predicted noise.

        Args:
            - latent : (1024x1024,)
            - diffusion_prior_grad : (1024x1024,), the predicted noise from the diffusion model
            - grad_method : 'start', 'step', 'noise_plus_denoise', 'euler'

        """

        if grad_method == "step":
            # update latent with the grad; directly update value
            latent.data = latent.data + diffusion_prior_grad

        elif grad_method == "start":
            # use normalized grad to update
            lr = kwargs["lr"]
            prior_weight = kwargs["prior_weight"]

            norm_latent_grad = torch.norm(latent.grad).item()
            norm_diffusion_prior_grad = torch.norm(diffusion_prior_grad).item()

            print(
                "Norm of Latent Grad:",
                norm_latent_grad,
                "Norm of Diffusion Prior Grad:",
                norm_diffusion_prior_grad,
            )

            norm_coeff = norm_latent_grad / norm_diffusion_prior_grad

            lr_prior = lr * norm_coeff * prior_weight

            # update latent with the grad; directly update value
            latent.data = latent.data + lr_prior * diffusion_prior_grad

        elif grad_method == "euler":
            # the grad already has scale.
            latent.data = latent.data + diffusion_prior_grad

        elif grad_method == "noise_plus_denoise":
            x_0_hat = latent.clone().detach()

            x_0_curve = diffusion_prior_grad
            grad_to_latent = x_0_hat - x_0_curve

            param = kwargs["prior_fusion_weight"]

            latent.data = latent.data - param * grad_to_latent

        else:
            raise NotImplementedError

        return latent

    def get_random_latent(self, sigma=None):
        """
        Init a random latent for the 1st step of the diffusion

        sigma: if None, use sigma_max as default.
        """

        # default param as shape-e
        steps, sigma_min, sigma_max, rho = (64, 0.001, 160, 7.0)
        device = "cuda"

        # sigma
        # sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
        # x_T = th.randn(*shape, device=device) * sigma_max

        if sigma is None:
            sigma = sigma_max

        # sample a random latent
        latent = torch.randn(1024 * 1024).to(device) * sigma

        return latent

    def get_clip_embedding_from_image(self, image, batch_size=1):
        model = self.model

        # load image into model_kwargs
        model_kwargs = dict(images=[image])

        if hasattr(model, "cached_model_kwargs"):
            model_kwargs = model.cached_model_kwargs(batch_size, model_kwargs)

        return model_kwargs["embeddings"]

    def get_latent_from_clip_embedding(
        self, clip_embed, batch_size=1, guidance_scale=3.0, clip_embedding_offset=None
    ):
        """
        @ clip_embed: (batch, 1024, 256)
        """

        # (Batchsize, 1024x1024)
        latents = sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(
                embeddings=clip_embed, clip_embedding_offset=clip_embedding_offset
            ),  # Update: Support CLIP feature offset
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        return latents[0]
