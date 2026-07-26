# Gesture-Controlled 3D Satellite Viewer

A prototype for touchless 3D model inspection and display control using webcam-based hand tracking.

The application captures live camera frames, detects hand landmarks with MediaPipe, and maps gesture input to 3D translation, rotation, and model selection for rendered STL geometry.

> This repository is an interactive demo and research prototype. It is not certified for flight, mission control, or safety-critical aerospace use.

## Key Features

- Webcam hand tracking using MediaPipe.
- 3D real-time STL rendering with ModernGL.
- Gesture controls for translation, rotation, and mode selection.
- Scene support for:
  - Curiosity rover
  - Saturn V launch vehicle
  - Nancy Grace Roman Space Telescope
- Saturn V assembly/explode view via gesture state.
- Hand gesture smoothing with hysteresis, debounce, dead zones, and a One Euro filter.

## Supported gestures

| Gesture | Result |
| --- | --- |
| Thumb + index pinch | Move object in X/Y, palm size changes depth |
| Thumb + index + middle pinch | Rotate object around X/Y |
| Open palm + wrist rotation | Cycle active model selection |
| Hold fist 3s | Confirm currently selected model |
| Short fist on Saturn V | Assemble stages |
| Two open hands on Saturn V | Trigger exploded view |
| `Q` key | Quit the app |

## Getting started

### Requirements

- Python 3.11
- Webcam
- GPU/driver capable of OpenGL 3.3 for ModernGL

### Install dependencies

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Run the viewer

```bash
python python/ar_cube_test.py --model rover
```

Valid model choices:
- `rover`
- `saturn`
- `roman`

### Run tests

```bash
python -m unittest discover -s tests
```

## Project structure

- `python/ar_cube_test.py` — main application, camera loop, gesture mapping, and rendering.
- `python/interaction_math.py` — independent gesture geometry helpers and filters.
- `python/models/` — STL assets for rendered scenes.
- `tests/` — unit tests for gesture math behavior.
- `docs/VALIDATION.md` — validation notes and known limitations.

## Behavior notes

- Gesture thresholds normalize by palm size so the system is less sensitive to hand distance from camera.
- Hysteresis and debounce are used to reduce flicker when gestures are near borderline values.
- The One Euro filter smooths small hand jitter while allowing faster motion to respond cleanly.
- Saturn V stage positions are computed from ordered STL parts with a shared scale so the vehicle assembles consistently.

## Troubleshooting

- If the webcam does not open, close other camera apps and retry.
- If rendering fails, confirm that OpenGL 3.3 or later is available on your system.
- If the model appears too small or too large, adjust the camera distance or window size.

## Notes

This repository is intended as a developer demo and experiment in gesture-driven 3D interaction. It is designed for local use and demonstration rather than deployment in production aerospace workflows.
