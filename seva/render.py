import glob
import os
import os.path as osp

import fire
import numpy as np
import torch
import torch.nn.functional as F

from seva.data_io import get_parser
from seva.eval import (
    IS_TORCH_NIGHTLY,
    create_transforms_simple,
    infer_prior_stats,
    run_one_scene,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.utils import load_model

device = "cuda:0"

if IS_TORCH_NIGHTLY:
    COMPILE = True
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
else:
    COMPILE = False

MODEL = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
AE = AutoEncoder(chunk_size=1).to(device)
CONDITIONER = CLIPConditioner().to(device)
DISCRETIZATION = DDPMDiscretization()
DENOISER = DiscreteDenoiser(discretization=DISCRETIZATION, num_idx=1000, device=device)
VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}

if COMPILE:
    MODEL = torch.compile(MODEL, dynamic=False)
    CONDITIONER = torch.compile(CONDITIONER, dynamic=False)
    AE = torch.compile(AE, dynamic=False)


def parse_task(
    scene,
    num_inputs,
    T,
    version_dict,
):
    options = version_dict["options"]

    anchor_indices = None
    anchor_c2ws = None
    anchor_Ks = None

    parser = get_parser(
        parser_type="reconfusion",
        data_dir=scene,
        normalize=False,
    )
    all_imgs_path = parser.image_paths
    c2ws = parser.camtoworlds
    camera_ids = parser.camera_ids
    Ks = np.concatenate([parser.Ks_dict[cam_id][None] for cam_id in camera_ids], 0)

    if num_inputs is None:
        assert len(parser.splits_per_num_input_frames.keys()) == 1
        num_inputs = list(parser.splits_per_num_input_frames.keys())[0]
        split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
    elif isinstance(num_inputs, str):
        split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
        num_inputs = int(num_inputs.split("-")[0])  # for example 1_from32
    else:
        split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore

    num_targets = len(split_dict["test_ids"])

    num_anchors = infer_prior_stats(
        T,
        num_inputs,
        num_total_frames=num_targets,
        version_dict=version_dict,
    )

    target_c2ws = c2ws[split_dict["test_ids"], :3]
    target_Ks = Ks[split_dict["test_ids"]]
    anchor_c2ws = target_c2ws[
        np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
    ]
    anchor_Ks = target_Ks[
        np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
    ]

    sampled_indices = split_dict["train_ids"] + split_dict["test_ids"]
    all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
    c2ws = c2ws[sampled_indices]
    Ks = Ks[sampled_indices]

    input_indices = np.arange(num_inputs).tolist()
    anchor_indices = np.linspace(
        num_inputs, num_inputs + num_targets - 1, num_anchors
    ).tolist()

    return (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        torch.tensor(c2ws[:, :3]).float(),
        torch.tensor(Ks).float(),
        (torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None),
        (torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None),
    )


def render_video(
    scene_input_dir: str,
    scene_output_dir: str,
    H=None,
    W=None,
    T=None,
    use_traj_prior=True,
    **overwrite_options,
):
    if H is not None:
        VERSION_DICT["H"] = H
    if W is not None:
        VERSION_DICT["W"] = W
    if T is not None:
        VERSION_DICT["T"] = [int(t) for t in T.split(",")] if isinstance(T, str) else T

    options = VERSION_DICT["options"]
    options["chunk_strategy"] = "interp"
    options["video_save_fps"] = 30.0
    options["beta_linear_start"] = 5e-6
    options["log_snr_shift"] = 2.4
    options["guider_types"] = 1
    # Settings based on https://github.com/Stability-AI/stable-virtual-camera/blob/main/docs/CLI_USAGE.md#img2trajvid
    options["cfg"] = [4.0, 2.0]
    options["camera_scale"] = 2.0
    options["num_steps"] = 50
    options["cfg_min"] = 1.2
    options["encoding_t"] = 1
    options["decoding_t"] = 1
    options["num_inputs"] = None
    options["seed"] = 23
    options.update(overwrite_options)

    num_inputs = options["num_inputs"]
    seed = options["seed"]

    save_path_scene = scene_output_dir

    # parse_task -> infer_prior_stats modifies VERSION_DICT["T"] in-place.
    (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        c2ws,
        Ks,
        anchor_c2ws,
        anchor_Ks,
    ) = parse_task(
        scene_input_dir,
        num_inputs,
        VERSION_DICT["T"],
        VERSION_DICT,
    )
    assert num_inputs is not None
    # Create image conditioning.
    image_cond = {
        "img": all_imgs_path,
        "input_indices": input_indices,
        "prior_indices": anchor_indices,
    }
    # Create camera conditioning.
    camera_cond = {
        "c2w": c2ws.clone(),
        "K": Ks.clone(),
        "input_indices": list(range(num_inputs + num_targets)),
    }
    # run_one_scene -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
    video_path_generator = run_one_scene(
        "img2trajvid",
        VERSION_DICT,  # H, W maybe updated in run_one_scene
        model=MODEL,
        ae=AE,
        conditioner=CONDITIONER,
        denoiser=DENOISER,
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=save_path_scene,
        use_traj_prior=use_traj_prior,
        traj_prior_Ks=anchor_Ks,
        traj_prior_c2ws=anchor_c2ws,
        seed=seed,  # to ensure sampled video can be reproduced in regardless of start and i
    )
    for _ in video_path_generator:
        pass

    # Convert from OpenCV to OpenGL camera format.
    c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()
    img_paths = sorted(glob.glob(osp.join(save_path_scene, "samples-rgb", "*.png")))
    if len(img_paths) != len(c2ws):
        input_img_paths = sorted(
            glob.glob(osp.join(save_path_scene, "input", "*.png"))
        )
        assert len(img_paths) == num_targets
        assert len(input_img_paths) == num_inputs
        assert c2ws.shape[0] == num_inputs + num_targets
        target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
        img_paths = [
            input_img_paths[input_indices.index(i)]
            if i in input_indices
            else img_paths[target_indices.index(i)]
            for i in range(c2ws.shape[0])
        ]
    create_transforms_simple(
        save_path=save_path_scene,
        img_paths=img_paths,
        img_whs=np.array([VERSION_DICT["W"], VERSION_DICT["H"]])[None].repeat(
            num_inputs + num_targets, 0
        ),
        c2ws=c2ws,
        Ks=Ks,
    )


# if __name__ == "__main__":
#     fire.Fire(main)
