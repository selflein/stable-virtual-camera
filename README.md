# Stable Virtual Camera

<a href="https://stable-virtual-camera.github.io"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project%20Page-gray.svg"></a>
<a href="http://arxiv.org/abs/2503.14489"><img src="https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2503.14489-B31B1B.svg"></a>
<a href="https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control"><img src="https://img.shields.io/badge/%F0%9F%93%83%20Blog-Stability%20AI-orange.svg"></a>
<a href="https://huggingface.co/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>
<a href="https://huggingface.co/spaces/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20Gradio%20Demo-Huggingface-orange"></a>
<a href="https://www.youtube.com/channel/UCLLlVDcS7nNenT_zzO3OPxQ"><img src="https://img.shields.io/badge/%F0%9F%8E%AC%20Video-YouTube-orange"></a>

`Stable Virtual Camera (Seva)` is a 1.3B generalist diffusion model for Novel View Synthesis (NVS), generating 3D consistent novel views of a scene, given any number of input views and target cameras.

# :tada: News

- March 2025 - `Stable Virtual Camera` is out everywhere.

# :wrench: Installation

To setup the virtual environment and install all necessary model dependencies, simply run:

```bash
pip install -e .
```

Check [INSTALL.md](docs/INSTALL.md) for other dependencies if you want to use our demos or develop from this repo.

# :open_book: Usage

We provide two demos for you to interative with `Stable Virtual Camera`.

### :rocket: Gradio demo

This gradio demo is a GUI interface that requires no expertised knowledge, suitable for general users. Simply run

```bash
python demo_gr.py
```

For a more detailed guide, follow [GR_USAGE.md](docs/GR_USAGE.md).

### :computer: CLI demo

This cli demo allows you to pass in more options and control the model in a fine-grained way, suitable for power users and academic researchers. An examplar command line looks as simple as

```bash
python demo.py --data_path <data_path> [additional arguments]
```

For a more detailed guide, follow [CLI_USAGE.md](docs/CLI_USAGE.md).

For users interested in benchmarking NVS models using command lines, check [`benchmark`](benchmark/) containing the details about scenes, splits, and input/target views we reported in the <a href="http://arxiv.org/abs/2503.14489">paper</a>.

# :books: Citing

If you find this repository useful, please consider giving a star :star: and citation.

```
@article{zhou2025stable,
    title={Stable Virtual Camera: Generative View Synthesis with Diffusion Models},
    author={Jensen (Jinghao) Zhou and Hang Gao and Vikram Voleti and Aaryaman Vasishta and Chun-Han Yao and Mark Boss and
    Philip Torr and Christian Rupprecht and Varun Jampani
    },
    journal={arXiv preprint},
    year={2025}
}
```
