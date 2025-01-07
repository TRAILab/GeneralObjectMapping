import argparse
import json
import os

from addict import Dict


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="path to config file", required=True
    )

    parser.add_argument(
        "-d",
        "--sequence_dir",
        type=str,
        default="data/KITTI/07",
        required=False,
        help="path to kitti sequence",
    )
    parser.add_argument("-i", "--frame_id", type=int, required=False, default=None, help="frame id")
    parser.add_argument(
        "-s",
        "--split_filename",
        type=str,
        required=False,
        help=".json split files for ShapeNetPreprocessed.",
    )
    parser.add_argument(
        "--data_source", type=str, required=False, help="data set path for ShapeNetPreprocessed."
    )
    parser.add_argument(
        "--dataset_source",
        type=str,
        required=False,
        help="choose type: pointcloud, deepsdf sampling etc.",
        default="deepsdf",
    )
    parser.add_argument("--loss_type", type=str, default="energy_score")
    parser.add_argument("--mask_method", type=str, required=False, default=None)
    parser.add_argument(
        "--num_iterations", type=int, required=False, default=100
    )  ### NOT USED, use iter_num instead ###
    parser.add_argument("--jump", type=bool, required=False, default=False)
    parser.add_argument("--cur_time", type=str, required=False, default=None)
    parser.add_argument(
        "--init_sigma",
        type=float,
        required=False,
        default=0.01,
        help="when using uncertainty reconstruction, define the sigma of initialized codes.",
    )
    parser.add_argument(
        "--prefix", type=str, required=False, default="deepsdf", help="The name of this exp."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=False,
        default=1,
        help="The random seed of the whole script.",
    )
    parser.add_argument(
        "--use_2d_loss", required=False, default=False, action="store_true", help="Use 2d loss."
    )

    # add sample_num
    parser.add_argument(
        "--sample_num",
        type=int,
        required=False,
        default=10,
        help="The number of sample for uncertainty.",
    )

    # add visualize_intermediate
    parser.add_argument(
        "--visualize_intermediate",
        required=False,
        default=False,
        action="store_true",
        help="Visualize intermediate results.",
    )

    # add bool vis_abs_uncer
    parser.add_argument(
        "--vis_abs_uncer",
        required=False,
        default=False,
        action="store_true",
        help="Visualize absolute uncertainty.",
    )

    # add for scannet dataset: args.scene_name, args.obj_id
    parser.add_argument(
        "--scene_name",
        type=str,
        required=False,
        default=None,
        help="The scene name of scannet dataset.",
    )
    parser.add_argument(
        "--obj_id", type=int, required=False, default=None, help="The object id of scannet dataset."
    )

    # dataset_name
    parser.add_argument(
        "--dataset_name", type=str, required=False, default="KITTI", help="The dataset name."
    )

    # save_root
    parser.add_argument(
        "--save_root",
        type=str,
        required=False,
        default="./output",
        help="The root dir to save results.",
    )

    # add loss_type_2d_uncertain
    parser.add_argument(
        "--loss_type_2d_uncertain",
        type=str,
        required=False,
        default="energy_score",
        help="The loss type for 2d loss.",
    )

    # whether to use gt_association
    parser.add_argument(
        "--use_gt_association",
        required=False,
        default=True,
        action="store_true",
        help="Use gt association.",
    )

    # add --dataset_subset_package, continue_from_scene
    parser.add_argument(
        "--dataset_subset_package",
        type=str,
        required=False,
        default=None,
        help="The dataset subset package.",
    )
    parser.add_argument(
        "--continue_from_scene",
        type=str,
        required=False,
        default=None,
        help="The scene name to continue from.",
    )
    parser.add_argument(
        "--continue_from_instance",
        type=int,
        required=False,
        default=None,
        help="The scene name to continue from.",
    )

    # consider view num; if 1, then single-view; if more than 1, multi-view
    parser.add_argument(
        "--view_num", type=int, required=False, default=1, help="The number of views to consider."
    )

    # a param to change init method, two options: gt_noise, estimator
    parser.add_argument(
        "--pose_init_method",
        type=str,
        required=False,
        default="estimator",
        help="The init method for pose.",
    )

    # add debug option
    parser.add_argument(
        "--debug", required=False, default=False, action="store_true", help="Debug mode."
    )

    # add close_2d_loss, close_3d_loss
    parser.add_argument(
        "--close_2d_loss", required=False, default=False, action="store_true", help="Close 2d loss."
    )
    parser.add_argument(
        "--close_3d_loss", required=False, default=False, action="store_true", help="Close 3d loss."
    )

    # add lr
    parser.add_argument("--lr", type=float, required=False, default=0.01, help="The learning rate.")

    # add init_sigma_pose, init_sigma_scale
    parser.add_argument(
        "--init_sigma_pose",
        type=float,
        required=False,
        default=0.01,
        help="The init sigma for pose.",
    )
    parser.add_argument(
        "--init_sigma_scale",
        type=float,
        required=False,
        default=0.01,
        help="The init sigma for scale.",
    )

    # add weight_3d
    parser.add_argument(
        "--weight_3d", type=float, required=False, default=100, help="The weight for 3d loss."
    )
    # add weight_2d
    parser.add_argument(
        "--weight_2d", type=float, required=False, default=50, help="The weight for 2d loss."
    )
    # add weight_norm
    parser.add_argument(
        "--weight_norm", type=float, required=False, default=500, help="The weight for norm loss."
    )

    # add open_visualization
    parser.add_argument(
        "--open_visualization",
        required=False,
        default=False,
        action="store_true",
        help="Open visualization.",
    )

    # add render_2d_K, default 400
    parser.add_argument(
        "--render_2d_K", type=int, required=False, default=400, help="The K for 2d rendering."
    )

    # add render_2d_calibrate_C, default 1.0
    parser.add_argument(
        "--render_2d_calibrate_C",
        type=float,
        required=False,
        default=1.0,
        help="The C for 2d rendering.",
    )

    # add evaluate_uncertainty_3d
    parser.add_argument(
        "--evaluate_uncertainty_3d",
        required=False,
        default=False,
        action="store_true",
        help="Evaluate uncertainty 3d.",
    )

    # add render_2d_const_a
    parser.add_argument(
        "--render_2d_const_a",
        type=float,
        required=False,
        default=0.0,
        help="The const a for 2d rendering.",
    )

    # add option_select_frame
    parser.add_argument("--option_select_frame", type=str, required=False, default="stage_3")

    # add loss_regularizer_method
    parser.add_argument(
        "--loss_regularizer_method",
        type=str,
        required=False,
        default="single",
        help="The method for loss regularizer.",
    )

    # add mask_source
    parser.add_argument(
        "--mask_source", type=str, required=False, default="gt", help="The source of mask."
    )

    # add cache_dir
    parser.add_argument(
        "--cache_dir", type=str, required=False, default="./output/cache", help="The cache dir."
    )

    # add mask_path_root
    parser.add_argument(
        "--mask_path_root",
        type=str,
        required=False,
        default=None,
        help="The mask dir. Used by Computecanada to avoid out of number files.",
    )

    # add diffusion_prior_valid_start_iter, default 0
    parser.add_argument(
        "--diffusion_prior_valid_start_iter",
        type=int,
        required=False,
        default=0,
        help="The start iter for diffusion prior.",
    )

    # add weight_latent_zero_norm, default 0
    parser.add_argument(
        "--weight_latent_zero_norm",
        type=float,
        required=False,
        default=0,
        help="The weight for latent zero norm loss.",
    )

    # add skip_optimization_after, default None
    parser.add_argument(
        "--skip_optimization_after",
        type=int,
        required=False,
        default=None,
        help="Skip optimization after.",
    )

    # add optimization_pose_clip_val, default 1e-9
    parser.add_argument(
        "--optimization_pose_clip_val",
        type=float,
        required=False,
        default=1e-9,
        help="The clip value for pose optimization.",
    )

    parser.add_argument(
        "--specific_category",
        type=str,
        required=False,
        default=None,
        help="If set, only consider this category and skip others.",
    )

    parser.add_argument(
        "--job_total_num", type=int, required=False, default=None, help="total cluster number"
    )

    parser.add_argument(
        "--job_current",
        type=int,
        required=False,
        default=None,
        help="current job number, starting from 0",
    )

    # add opt_num_per_diffuse
    parser.add_argument("--opt_num_per_diffuse", type=int, required=False, default=1)

    # lr_scheduler_type
    parser.add_argument(
        "--lr_scheduler_type", type=str, required=False, default="none", help="lr_scheduler_type"
    )

    parser.add_argument(
        "--jobs_num", type=int, required=False, default=None, help="total cluster number"
    )
    parser.add_argument(
        "--job_id",
        type=int,
        required=False,
        default=None,
        help="current job number, starting from 0",
    )

    # icp_model_source
    parser.add_argument(
        "--icp_model_source", type=str, required=False, default="ours", help="icp_model_source"
    )

    # save_detailed_output
    parser.add_argument(
        "--save_detailed_output",
        required=False,
        default=False,
        action="store_true",
        help="Whether to keep history.",
    )

    # prior_fusion_weight
    parser.add_argument(
        "--prior_fusion_weight",
        type=float,
        required=False,
        default=2e-1,
        help="The weight for latent zero norm loss.",
    )

    # loss_type_2d_render, default mse
    parser.add_argument(
        "--loss_type_2d_render",
        type=str,
        required=False,
        default="mse",
        help="The loss type for 2d render.",
    )

    # co3d_subset_name
    parser.add_argument(
        "--co3d_subset_name",
        type=str,
        required=False,
        default="manyview_dev_0",
        help="The co3d subset name.",
    )

    # text_condition_given
    parser.add_argument(
        "--text_condition_given",
        type=str,
        required=False,
        default=None,
        help="The text condition given.",
    )

    # Add config for skipping existing results
    parser.add_argument(
        "--skip", required=False, default=False, action="store_true", help="Skip existing."
    )

    return parser


def config_parser_exp_param():
    """
    Temp params for experiments submitted to Clusters.
    """
    parser = argparse.ArgumentParser()

    # lr
    parser.add_argument("--lr", type=float, required=False, default=None, help="The learning rate.")

    # save_root
    parser.add_argument(
        "--save_root", type=str, required=False, default=None, help="The root dir to save results."
    )

    # shape_lr_balance, a float
    parser.add_argument(
        "--shape_lr_balance",
        type=float,
        required=False,
        default=None,
        help="The balance between shape and pose.",
    )

    # iter_num
    parser.add_argument(
        "--iter_num", type=int, required=False, default=None, help="The number of iterations."
    )

    # view_num
    parser.add_argument(
        "--view_num",
        type=int,
        required=False,
        default=None,
        help="The number of views to consider.",
    )

    # sequence_dir
    parser.add_argument(
        "--sequence_dir", type=str, required=False, default=None, help="path to dataset sequence"
    )

    # cache_dir
    parser.add_argument(
        "--cache_dir", type=str, required=False, default=None, help="The cache dir."
    )

    # add mask_path_root
    parser.add_argument(
        "--mask_path_root",
        type=str,
        required=False,
        default=None,
        help="The mask dir. Used by Computecanada to avoid out of number files.",
    )

    # add vis_jump: int type
    parser.add_argument(
        "--vis_jump", type=int, required=False, default=None, help="The jump for visualization."
    )

    # add parameters for optimization, geometric_constraints_w_depth, geometric_constraints_w_2d, geometric_constraints_w_3d
    # add geometric_constraints_w_depth
    parser.add_argument(
        "--geometric_constraints_w_depth",
        type=float,
        required=False,
        default=None,
        help="Weight for geometric constraints in depth.",
    )
    # add geometric_constraints_w_2d
    parser.add_argument(
        "--geometric_constraints_w_2d",
        type=float,
        required=False,
        default=None,
        help="Weight for geometric constraints in 2D.",
    )
    # add geometric_constraints_w_3d
    parser.add_argument(
        "--geometric_constraints_w_3d",
        type=float,
        required=False,
        default=None,
        help="Weight for geometric constraints in 3D.",
    )

    # add method_init_pose
    parser.add_argument(
        "--method_init_pose",
        type=str,
        required=False,
        default=None,
        help="The method for pose initialization.",
    )

    # add method_init_pose_noise_level
    parser.add_argument(
        "--method_init_pose_noise_level",
        type=float,
        required=False,
        default=None,
        help="The noise level for pose initialization.",
    )

    # add optimization_pose_clip_val
    parser.add_argument(
        "--optimization_pose_clip_val",
        type=float,
        required=False,
        default=None,
        help="The clip value for pose optimization.",
    )

    # add specific_category
    parser.add_argument(
        "--specific_category",
        type=str,
        required=False,
        default=None,
        help="If set, only consider this category and skip others.",
    )

    parser.add_argument(
        "--scene_name",
        type=str,
        required=False,
        default=None,
        help="The scene name of scannet dataset.",
    )

    # add visualize_frames
    parser.add_argument(
        "--visualize_frames", required=False, default=None, action="store_true", help="Save Images"
    )

    # add param for sub-jobs on clusters
    parser.add_argument(
        "--jobs_num", type=int, required=False, default=None, help="total cluster number"
    )

    parser.add_argument(
        "--job_id",
        type=int,
        required=False,
        default=None,
        help="current job number, starting from 0",
    )

    parser.add_argument("--opt_num_per_diffuse", type=int, required=False, default=None)

    parser.add_argument(
        "--diffusion_prior_valid_start_iter",
        type=int,
        required=False,
        default=None,
        help="The start iter for diffusion prior.",
    )

    parser.add_argument(
        "--lr_scheduler_type", type=str, required=False, default=None, help="lr_scheduler_type"
    )

    parser.add_argument("--init_latent_method", type=str, required=False, default=None)

    parser.add_argument(
        "--icp_model_source", type=str, required=False, default=None, help="icp_model_source"
    )

    parser.add_argument("--grad_method", type=str, required=False, default=None)

    # dataset_category
    parser.add_argument(
        "--dataset_category", type=str, required=False, default=None, help="The dataset category."
    )

    # diffusion_condition_type
    parser.add_argument(
        "--diffusion_condition_type",
        type=str,
        required=False,
        default=None,
        help="The diffusion condition type.",
    )

    return parser


class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        # raise KeyError(key)
        print("miss key: ", key)


def get_configs(cfg_file, with_comment=False):
    with open(cfg_file) as f:
        if with_comment:
            import commentjson

            cfg_dict = commentjson.load(f)
        else:
            cfg_dict = json.load(f)
    return ForceKeyErrorDict(**cfg_dict)


def load_args():
    """
    Args Processing

    Source 1: Default defined parameters of add_arguments
     - Can Be Temperally overwrite parameters from command line

    Source 2: Load parameters defined in config file

    """
    # Source 1
    parser = config_parser()
    args, unknown = parser.parse_known_args()

    # Source 2: load from args.config, a json file
    config_file = args.config
    if config_file is not None:
        if os.path.isfile(config_file):
            configs = get_configs(config_file, with_comment=True)
            # merge configs into args
            args_vars = vars(args)
            args_vars.update(configs)

            args = ForceKeyErrorDict(**args_vars)

            print("Update param from config: ", config_file)
        else:
            raise FileExistsError

    # Source 3: Extra specified parameters, if set, update it
    # TODO: Add a temp config_parser to update debug params
    parser_exp = config_parser_exp_param()
    args_exp, unknown = parser_exp.parse_known_args()
    # if not None, update args
    for arg_exp in vars(args_exp):
        if getattr(args_exp, arg_exp) is not None:
            setattr(args, arg_exp, getattr(args_exp, arg_exp))
            print("Update param from exp: ", arg_exp, getattr(args_exp, arg_exp))

    return args
