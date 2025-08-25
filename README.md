# YOLO-CL: YOLO for CLuster detection

**YOLO-CL** (YOLO for CLuster detection) is a TensorFlow-based deep learning framework that extends the YOLOv3 object detector to astronomical cluster finding.  It addresses the problem of **galaxy cluster detection** in wide-field survey images by training a convolutional neural network to locate clusters directly from raw sky images.  By operating on the multi-band images themselves, YOLO-CL learns the photometric patterns of clusters without relying on intermediate catalogs of individual galaxies.  The method has been demonstrated on Sloan Digital Sky Survey (SDSS) images and simulated Vera Rubin Observatory (LSST) data, achieving high completeness and purity.  For example, on LSST's DC2 mock data, YOLO-CL recovers ≥94% of massive clusters (M > 10^14 M☉) across redshifts z<1, outperforming traditional algorithms.  In summary, YOLO-CL contributes a **novel cluster detection benchmark and model** that delivers fast, survey-scale cluster catalogs with well-defined selection functions.

## Key Features

- **YOLOv3-based architecture:** YOLO-CL uses the standard YOLOv3 backbone (Darknet-53) for multi-scale object detection.  This includes the full suite of convolutional, residual, and upsampling layers to detect clusters of various apparent sizes in a single pass.
- **Direct image input:** The network works on raw survey images in the usual image coordinate frame.  It does **not** require precomputed object catalogs or photometric redshifts, removing common sources of systematic error.  In effect, YOLO-CL "sees" the clusters as bright extended features in the image.
- **High completeness/purity:** In tests on SDSS and LSST simulations, YOLO-CL attains very high recovery rates.  For example, the DC2 cluster catalog is 100% complete at halo masses M₅₀₀>10^14.6 M☉ (z<0.8) and 94% complete at M₅₀₀>10^14 M☉ (z≲1) , with only ~6% false positives.  This performance rivals or exceeds traditional matched-filter and SZ/X-ray methods.
- **Configurable anchors and scales:** The repository includes presets for baseline anchor boxes and allows custom anchors (see `runs/baseline_anchors.txt`).  You can adjust the input size (`cfg.YOLO.SIZE`) and anchor strategy to match your data (e.g. 1024x2048 px images).
- **Versatile dataset support:** Although developed for clusters from SDSS and LSST simulations, YOLO-CL can be trained on any image dataset in the proper YOLO format (see *Dataset Preparation* below).

## Installation

1. **Clone the repository.** In a terminal, run:
   ```bash
   git clone https://github.com/s-ilic/YOLO-CL.git
   cd YOLO-CL
   ```
2. **Set up the environment.** YOLO-CL requires **Python 3** and TensorFlow (the code is tested with TensorFlow 2.x). Install the core packages via pip:
   ```bash
   pip install tensorflow easydict
   ```
   Optionally install other common ML tools as needed (e.g. `numpy`, `opencv-python`, etc.).
3. **GPU support (optional).** For acceleration, ensure CUDA/cuDNN are installed and use the GPU-enabled TensorFlow. Otherwise YOLO-CL will run in CPU mode (much slower).
4. **Verify installation.** You should be able to import YOLO-CL modules in Python, e.g.:
   ```python
   import core.yolov3, core.dataset, core.config
   ```
   If there are errors, check that all packages are correctly installed.

## Dataset Preparation

YOLO-CL expects training and test data in the *YOLO format*: a directory of images plus annotation text files listing bounding boxes. Each line of a `.txt` label file should have `xmin,ymin,xmax,ymax,class_id` for an object in the corresponding image. The repository includes a template `runs/clusters.names` (class list) and expects a “train.txt” file (under `runs/<experiment>/train.txt`) listing each training image and its boxes.

In general, you can use any image dataset by converting its annotations to YOLO format. Place images in a directory (e.g. `images/`), labels in `labels/`, and point the config to your lists. Ensure `core/config.py` has the correct `CLASSES` file and path to your train/test annotation lists.

The key is that YOLO-CL reads a text file where each line has: `path/to/image.jpg xmin,ymin,xmax,ymax,class`. All objects (clusters or other) in that image should be on the same line. See example files (such as `data/dataset/yymnist_test.txt`) for formatting.

## Training and Evaluation

Once your data is prepared, train a YOLO-CL model with:
```bash
python train.py
```
This will read the parameters from `core/config.py` (check `cfg.YOLO.ROOT` for the experiment name and the train annotation path). During training, loss values will print to the console (you can toggle verbosity) and the model weights will be saved periodically under `runs/<ROOT>/` (e.g. `yolov3.ckpt`). The script will by default train until the learning rate schedule completes (similar to standard YOLOv3 training).

To run inference or evaluate on a test set, use:
```bash
python test.py
```
This script loads the latest or specified model checkpoint from `runs/<ROOT>/` and applies it to images. You can configure it to output predicted bounding boxes on a set of images. (For example, modifying the code to save annotated output or to compare against ground-truth boxes.) Expected outcomes are detection outputs in the YOLO result format and printed metrics such as mAP or precision/recall for your test data.

Here are example usage steps:

- Edit `core/config.py` for your experiment: set `cfg.YOLO.CLASSES` (to match your classes), `cfg.TRAIN.ANNOT_PATH` and `cfg.TEST.ANNOT_PATH` (pointing to your train/val lists), and other parameters (image size, batch size, etc.).
- Run training: `python train.py`. You should see logs like loss values and learning rate updates. Training may take minutes to hours depending on dataset and hardware.
- After training, run evaluation on validation images: `python test.py`. This will print detection results. You can also modify `test.py` to save annotated images or compute specific statistics.

In our experiments on SDSS and LSST DC2 data, training yielded models that locate clusters effectively. You can plot training/validation losses or run TensorBoard if `DO_TBOARD=True` in the config.

## Pretrained Models

At present, **no pretrained model weights are included** in this repository. Users should train a model on their target dataset. (The authors' trained checkpoints for SDSS and LSST tests may be available on request.) Once you train your own model, you can reuse the resulting checkpoint for inference or further fine-tuning. We plan to add downloadable model files in the future if possible.

## Reproducibility and Configuration

YOLO-CL is fully configurable via `core/config.py`, which uses an EasyDict (`edict`) to hold all settings. Key configurations include:

- **Experiment name:** `cfg.YOLO.ROOT` sets the output folder under `runs/`. Use a unique name to separate runs (e.g. “voc2012_run”).
- **Class names:** Update `cfg.YOLO.CLASSES` to point to a text file listing class names (one per line). For example, for cluster detection use `runs/clusters.names`.
- **Anchors and scales:** The default anchors are in `runs/baseline_anchors.txt`. You can generate custom anchors for your objects and set `cfg.YOLO.ANCHORS`. Change `cfg.YOLO.STRIDES` and `cfg.YOLO.ANCHOR_PER_SCALE` if needed.
- **Training settings:** Under `__C.TRAIN`, set `VERBOSE`, `DO_TBOARD`, `TRAIN_BATCH_SIZE`, `TEST_BATCH_SIZE`, and the annotation paths `ANNOT_PATH` (pointing to your dataset lists). Random seeds are not fixed by default, so for exact reproducibility you may want to set `numpy` and `tensorflow` seeds manually in the scripts.
- **Input size:** Change `cfg.YOLO.SIZE` to match your image resolution (e.g. 512 or 1024). This defines the neural network input layer.
- **Hardware & software:** Note your software versions (TensorFlow, CUDA) to reproduce results. Logging all relevant config values (or saving the `config.py` used) helps keep track.

By keeping a record of the config file and random seeds, you can reproduce any experiment. For consistent benchmarking (as in the reference papers, ensure you use the same data splits and parameter settings.

## Citations

If you use YOLO-CL or its ideas, please cite the following works:

- [[2301.09657] YOLO-CL: Galaxy cluster detection in the SDSS with deep machine learning](https://arxiv.org/abs/2301.09657).
- [[2409.03333] YOLO-CL cluster detection in the Rubin/LSST DC2 simulation](https://arxiv.org/abs/2409.03333).

These references describe the algorithm and its performance in detail.

## License and Contribution Guidelines

This repository is released under the **MIT License** (see the `LICENSE` file or contact the authors if not present). You are free to use, modify, and distribute the code under that license.

Contributions are welcome! Please feel free to **open an issue or submit a pull request** on GitHub if you find bugs, request features, or have improvements.  For example, you might contribute scripts for converting new datasets into YOLO-CL format, add support for model checkpoints, or improve documentation. We ask that you follow the existing code style and include appropriate citations if you add new methods.

For any questions or suggestions, please use the GitHub repository's Issues page or contact the maintainers via email.