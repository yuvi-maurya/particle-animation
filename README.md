# 3D Model Box

Webcam-based Python/OpenCV demo using MediaPipe Hands and real OBJ meshes.

## Asset Structure

Selectable objects are OBJ source packages in `assets/`. The current loader chooses the newest valid source candidate for each intended model name, so updated files in `assets/` win over stale duplicates in `assets/models/`.

Expected models:

- `dragon.obj`
- `tree.obj`
- `flowers.obj`
- `butterfly.obj`

Each OBJ may reference an MTL file and texture images with relative paths. MTL diffuse colours are preserved. If a material has UV texture coordinates and a texture map, the renderer samples that texture once at load time to bake per-face colours and cutout alpha for fast OpenCV rendering.

## Cache Behavior

No persistent mesh cache is used at runtime. Models are parsed and prepared once when the app starts. A SHA-256 source fingerprint is computed from the OBJ, referenced MTL files, referenced textures, model configuration and cache schema version. Generated orientation renders and future cache/output folders are ignored by Git; source OBJ/MTL/texture files remain trackable.

## Behavior

- Physical left thumb/index wide separation shows the yellow 3D box and selected OBJ model from the midpoint between the two fingertips.
- Physical left thumb-index pinch hides the cube/model.
- While visible, the cube/model stays centered between the left thumb and index fingertip and grows smoothly from tiny to full size when shown.
- Physical right palm OPEN switches to the next OBJ model once.
- Holding the palm open does not repeat. Close the fist to re-arm, then open again to switch again.
- Model switches use a shard, smoke, and emergence transition.
- The box and model share one continuous upright Y-axis 360 degree rotation.
- The OpenCV window starts fullscreen by default.
- Default camera view is clean: no text overlays or hand landmarks unless toggled.

## Controls

- `N`: next OBJ model
- `B`: previous OBJ model
- `H`: swap handedness mapping
- `D`: demo mode without hands
- `I`: debug overlay
- `L`: hand landmarks, off by default
- `Q` or `Esc`: quit

## Install

Python 3.10 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Camera backend selection:

```bash
python main.py --camera-backend msmf
python main.py --camera-backend default
python main.py --camera-backend dshow
```

Camera-only diagnostics without loading models or MediaPipe:

```bash
python main.py --camera-test --camera-backend auto
python main.py --camera-test --camera-backend msmf
python main.py --camera-test --camera-backend default
python main.py --camera-test --camera-backend dshow
```

By default the app uses `--camera-backend auto`, which tries MSMF first and then OpenCV's default backend. DirectShow is not selected automatically because some cameras return corrupted black/noisy frames through DSHOW.

## Validation

```bash
python main.py --self-test
python main.py --benchmark
```

## Troubleshooting

If a model is missing or invalid, the app prints the exact path and reason in the terminal and continues with other valid models. If textures are missing or cannot be decoded, the app prints a package warning and falls back to MTL diffuse colours for those faces. The software renderer uses cached per-face texture and cutout-alpha baking rather than full per-pixel UV rasterization.

If the camera opens to a black or corrupted frame, try `--camera-backend msmf` or `--camera-backend default`. Use `--camera-backend dshow` only when you specifically need DirectShow. The camera runs on a background capture thread so the UI can stay responsive if a backend stalls.

Manual webcam checks still required after automated tests:

- left hand shows and moves the model while visible;
- left thumb-index pinch hides the model;
- left thumb/index wide separation shows the model again;
- right palm open changes exactly one model;
- holding the palm open does not repeat;
- closing the fist re-arms without changing the model;
- opening again changes exactly once;
- `H` fixes reversed handedness if needed;
- default frame remains text-free and landmark-free.
