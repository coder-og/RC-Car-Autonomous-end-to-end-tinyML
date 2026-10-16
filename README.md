# Raspberry Pi RC Car — Learned Steering

A camera-based RC car project using a PilotNet-style CNN to predict steering. **Throttle is controlled manually** with a gamepad; this is not autonomous navigation or obstacle avoidance. Training runs on a computer and LiteRT inference runs on a Raspberry Pi.

## Pipeline

`data_collection.py` → JPEG frames + CSV labels → `test_crop.py` → `train.py` → `.tflite` → `validation.py` → `autonomous_drive.py`.

## Hardware and wiring used by the code

| Connection | BCM GPIO / setting |
|---|---|
| Steering servo signal | 18; pulse range 1300–1900 microseconds |
| Motor driver forward | 5 |
| Motor driver backward | 6 |
| Motor driver enable/PWM | 19 |
| Camera | Picamera2-supported Pi camera; 360 × 240 |
| Controller | pygame joystick 0; steering axis 0, forward axis 5, reverse axis 2 |

These are GPIO signal assignments, not a complete power schematic. Use an appropriate motor driver and separate motor supply with a common ground. Confirm steering direction and button mapping with wheels lifted before driving. The code uses `PiGPIOFactory` and requires a running `pigpiod` service on a compatible Raspberry Pi OS/Pi combination.

## Raspberry Pi setup

On Raspberry Pi OS with the camera enabled:

```sh
sudo apt update
sudo apt install python3-venv python3-picamera2 python3-opencv python3-numpy pigpio
sudo systemctl enable --now pigpiod
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install -r requirements-pi.txt
python data_collection.py
```

Package availability depends on the OS/architecture. The dependency files give compatibility ranges, not a tested device lock. Collection uses three processes (camera, controls, writer). Button **index 2** toggles recording; Ctrl+C exits. Driving uses button **index 0** to stop. The old comments called both buttons X; check the mapping for your controller.

## Dataset and training

Each recording creates `dataset/run_001/images/frame_000001.jpg` and `dataset/run_001/labels.csv` (then run_002, etc.). CSV columns:

`frame,timestamp,steering,throttle,smooth_steer,smooth_throttle`

Frames with low smoothed throttle are skipped. Transfer complete runs to the computer. Keep a separate held-out recording, for example `run_006_val/`, and set its path in `params.py`. The training loader also excludes that exact resolved directory if it is inside the dataset root. Avoid duplicated runs across training and validation.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-train.txt
python test_crop.py
python train.py
python validation.py
```

`test_crop.py` is an interactive crop viewer, not an automated test. Change `params.py` after selecting crop bounds. Current defaults are rows **60:140**, resized to **68 × 68 × 3**, with `smooth_steer` as the target. The model rescales 0–255 internally. Collection swaps camera channels before writing JPEGs; live inference now applies that same swap to match images loaded through OpenCV. Recheck the full camera-to-file channel convention when changing camera formats.

Training uses brightness/blur augmentation, steering-sign-correct horizontal flips, AdamW, weighted Huber loss, early stopping and a held-out validation run. All training arrays are loaded into RAM. The model basename is shared through `params.py`: `model_recommended_final_umar`. Training writes `.h5`, `.tflite`, and a second `.tflite` copy under `models/`.

## Driving

Copy `model_recommended_final_umar.tflite` to this project directory on the Pi, alongside `params.py` and `preprocessing.py`:

```sh
python autonomous_drive.py
```

Steering is smoothed and clamped to [-1, 1]. Manual throttle is capped at 0.4 in the current scripts. Keep an operator at the controls. The inference path expects the floating-point input model produced by this training flow; arbitrary integer-quantized models need explicit quantization handling.

## Verification and results

```sh
python -m compileall -q .
python -m unittest discover -s tests -v
```

Tests check crop validation, image shape/channel preservation, held-out-run exclusion and per-sample loss weighting. They do not operate GPIO or a camera. CI performs these checks and constructs the training model.

`validation.py` prints raw/smoothed MSE, correlation, and left/right/straight MSE, then plots predictions. No trained weights, dataset, measured FPS, or reproducible benchmark result is included here. Add real measurements with Pi model, OS, package versions, dataset split, sample count and model size; do not substitute training scores for unseen-road performance.

Known limitations: no obstacle detection, hardware disconnect watchdog or complete data-collection worker-failure recovery; correlation is undefined for constant signals; smoothing and camera changes require a new evaluation. Earlier trained models should be reevaluated after the preprocessing/loss corrections.
