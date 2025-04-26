import torch
import imageio.v3 as iio
import numpy as np
from einops import rearrange
from typing import Literal
import os
from seva.eval import (
    create_transforms_simple,
    transform_img_and_K,
)
import json
import os.path as osp

from seva.geometry import (
    DEFAULT_FOV_RAD,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_spiral_horizontal_w2cs,
    normalize_scene,
)


def get_dust3r(device: str):
    from seva.modules.preprocessor import Dust3rPipeline

    return Dust3rPipeline(device=device)  # type: ignore


def preprocess(
    input_img_path_or_tuples: list[str] | str, device: str = "cuda:0"
) -> dict:
    # Simply hardcode these such that aspect ratio is always kept and
    # shorter side is resized to 576. This is only to make GUI option fewer
    # though, changing it still works.
    shorter: int = 576
    # Has to be 64 multiple for the network.
    shorter = round(shorter / 64) * 64

    if isinstance(input_img_path_or_tuples, str):
        input_img_path_or_tuples = [input_img_path_or_tuples]

    img_paths = input_img_path_or_tuples
    (
        input_imgs,
        input_Ks,
        input_c2ws,
        points,
        point_colors,
    ) = get_dust3r(device).infer_cameras_and_points(img_paths, min_conf_thr=4.0)
    num_inputs = len(img_paths)
    if num_inputs == 1:
        input_imgs, input_Ks, input_c2ws, points, point_colors = (
            input_imgs[:1],
            input_Ks[:1],
            input_c2ws[:1],
            points[:1],
            point_colors[:1],
        )
    input_imgs = [img[..., :3] for img in input_imgs]
    # Normalize the scene.
    point_chunks = [p.shape[0] for p in points]
    point_indices = np.cumsum(point_chunks)[:-1]
    input_c2ws, points, _ = normalize_scene(  # type: ignore
        input_c2ws,
        np.concatenate(points, 0),
        camera_center_method="poses",
    )
    points = np.split(points, point_indices, 0)
    # Scale camera and points for viewport visualization.
    # scene_scale = np.median(
    #     np.ptp(np.concatenate([input_c2ws[:, :3, 3], *points], 0), -1)
    # )
    scene_scale = np.median(np.ptp(np.concatenate(points, 0), -1))
    input_c2ws[:, :3, 3] /= scene_scale
    points = [point / scene_scale for point in points]
    input_imgs = [
        torch.as_tensor(img / 255.0, dtype=torch.float32) for img in input_imgs
    ]
    input_Ks = torch.as_tensor(input_Ks)
    input_c2ws = torch.as_tensor(input_c2ws)
    new_input_imgs, new_input_Ks = [], []
    for img, K in zip(input_imgs, input_Ks):
        img = rearrange(img, "h w c -> 1 c h w")
        # If you don't want to keep aspect ratio and want to always center crop, use this:
        # img, K = transform_img_and_K(img, (shorter, shorter), K=K[None])
        img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
        assert isinstance(K, torch.Tensor)
        K = K / K.new_tensor([img.shape[-1], img.shape[-2], 1])[:, None]
        new_input_imgs.append(img)
        new_input_Ks.append(K)
    input_imgs = torch.cat(new_input_imgs, 0)
    input_imgs = rearrange(input_imgs, "b c h w -> b h w c")[..., :3]
    input_Ks = torch.cat(new_input_Ks, 0)
    return {
        "input_imgs": input_imgs,
        "input_Ks": input_Ks,
        "input_c2ws": input_c2ws,
        "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
        "points": points,
        "point_colors": point_colors,
        "scene_scale": scene_scale,
    }


def get_preset_cam_traj(
    preset_traj: Literal["orbit", "hemisphere"],
    start_w2c: torch.Tensor,
    look_at: torch.Tensor,
    up_direction: torch.Tensor,
    num_frames: int,
    fov: float,
    **kwargs,
):
    fovs = np.full((num_frames,), fov)
    if preset_traj == "orbit":
        poses = torch.linalg.inv(
            get_arc_horizontal_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames=num_frames,
                endpoint=False,
                ref_radius_scale=1.3,
                ref_up_shift=0.8,
                **kwargs,
            )
        ).numpy()
        return poses, fovs

    if preset_traj == "hemisphere":
        poses = torch.linalg.inv(
            get_spiral_horizontal_w2cs(
                start_w2c,
                look_at,
                up_direction,
                num_frames=num_frames,
                truncate_traj_ratio=0.5,
                ref_radius_scale=1.5,
                degree=360 * 3,
                **kwargs,
            )
        ).numpy()
        return poses, fovs
    raise ValueError(f"Preset trajectory {preset_traj} not supported")


def get_target_c2ws_and_Ks_from_preset(
    preprocessed: dict,
    preset_traj: Literal["orbit", "hemisphere"],
    num_frames: int,
    look_at: tuple[float, float, float] | None = (0, 0, 10),
    **kwargs,
):
    img_wh = preprocessed["input_wh"]
    start_c2w = preprocessed["input_c2ws"][0]
    start_w2c = torch.linalg.inv(start_c2w)
    look_at = torch.tensor(look_at)
    start_fov = DEFAULT_FOV_RAD
    target_c2ws, target_fovs = get_preset_cam_traj(
        preset_traj,
        num_frames=num_frames,
        start_w2c=start_w2c,
        look_at=look_at,
        up_direction=-start_c2w[:3, 1],
        fov=start_fov,
        **kwargs,
    )
    target_c2ws = torch.as_tensor(target_c2ws)
    target_fovs = torch.as_tensor(target_fovs)
    target_Ks = get_default_intrinsics(
        target_fovs,  # type: ignore
        aspect_ratio=img_wh[0] / img_wh[1],
    )
    return target_c2ws, target_Ks


def store_data_for_render(
    preprocessed: dict,
    target_c2ws: torch.Tensor,
    target_Ks: torch.Tensor,
    output_dir: str,
):
    input_imgs, input_Ks, input_c2ws, input_wh = (
        preprocessed["input_imgs"],
        preprocessed["input_Ks"],
        preprocessed["input_c2ws"],
        preprocessed["input_wh"],
    )

    num_inputs = len(input_imgs)
    num_targets = len(target_c2ws)

    input_imgs = (input_imgs.cpu().numpy() * 255.0).astype(np.uint8)
    input_c2ws = input_c2ws.cpu().numpy()
    input_Ks = input_Ks.cpu().numpy()
    target_c2ws = target_c2ws.cpu().numpy()
    target_Ks = target_Ks.cpu().numpy()
    img_whs = np.array(input_wh)[None].repeat(len(input_imgs) + len(target_Ks), 0)

    os.makedirs(output_dir, exist_ok=True)
    img_paths = []
    for i, img in enumerate(input_imgs):
        iio.imwrite(img_path := osp.join(output_dir, f"{i:03d}.png"), img)
        img_paths.append(img_path)
    for i in range(num_targets):
        iio.imwrite(
            img_path := osp.join(output_dir, f"{i + num_inputs:03d}.png"),
            np.zeros((input_wh[1], input_wh[0], 3), dtype=np.uint8),
        )
        img_paths.append(img_path)

    # Convert from OpenCV to OpenGL camera format.
    all_c2ws = np.concatenate([input_c2ws, target_c2ws])
    all_Ks = np.concatenate([input_Ks, target_Ks])
    all_c2ws = all_c2ws @ np.diag([1, -1, -1, 1])
    create_transforms_simple(output_dir, img_paths, img_whs, all_c2ws, all_Ks)
    split_dict = {
        "train_ids": list(range(num_inputs)),
        "test_ids": list(range(num_inputs, num_inputs + num_targets)),
    }
    with open(osp.join(output_dir, f"train_test_split_{num_inputs}.json"), "w") as f:
        json.dump(split_dict, f, indent=4)
