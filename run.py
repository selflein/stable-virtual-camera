"""Generate a video of a scene given sparse input views.

Creates a folder with the following structure:
```
├── first-pass  # Keyframes
│   ├── samples-rgb
│   │   ├── 000.png
│   │   ├── ...
│   │   └── 017.png
│   └── samples-rgb.mp4
├── input      # Input images
│   ├── 000.png
│   ├── ...
│   └── 002.png
├── input.mp4
├── samples-rgb  # Generated views
│   ├── 000.png
│   ├── 001.png
│   ├── ...
│   └── 059.png
├── samples-rgb.mp4
└── transforms.json  # Intrinsics/extrinsics for input images and generated views
```

"""

import numpy as np

from seva.render import render_video
from seva.preproc import (
    preprocess,
    get_target_c2ws_and_Ks_from_preset,
    store_data_for_render,
)


if __name__ == "__main__":
    work_dir = "/tmp/pillow_seva_input"
    output_dir = "/tmp/pillow_seva_output_no_traj"

    imgs = [
        "/data/zs-physics/physics-IQ-benchmark/switch-frames/0193_switch-frames_anyFPS_perspective-left_trimmed-weight-on-pillow.jpg",
        "/data/zs-physics/physics-IQ-benchmark/switch-frames/0194_switch-frames_anyFPS_perspective-center_trimmed-weight-on-pillow.jpg",
        "/data/zs-physics/physics-IQ-benchmark/switch-frames/0195_switch-frames_anyFPS_perspective-right_trimmed-weight-on-pillow.jpg",
    ]

    preproc_data = preprocess(imgs, device="cuda:0")

    # Compute the median of the point cloud as the reference point to look at.
    points = preproc_data["points"][0]
    median_point = np.median(points, axis=0)

    # Generate a spiraling camera trajectory around the center point.
    target_c2w, target_K = get_target_c2ws_and_Ks_from_preset(
        preproc_data,
        preset_traj="orbit",
        num_frames=60,
        look_at=median_point,
    )

    store_data_for_render(preproc_data, target_c2w, target_K, output_dir=work_dir)

    W, H = preproc_data["input_wh"]
    render_video(
        work_dir,
        output_dir,
        H=H,
        W=W,
        # Whether to generate a sequential video of frames (True) or frames conditioned on the closes ground truth views
        # `use_traj_prior=True` usually gives more geometrically consistent results in our experiments
        use_traj_prior=True,
    )
