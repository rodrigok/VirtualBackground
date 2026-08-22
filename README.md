# VirtualBackground Lab

A dark, local React workbench for testing real-time virtual-background processing. It runs selectable MediaPipe person-segmentation models in the browser and exposes the complete mask and compositing pipeline for experimentation.

## Features

- Image, four-up static benchmark grid, video, uploaded file, and live webcam inputs
- Google Selfie Multiclass as the quality-oriented default, plus Selfie Landscape and Pose comparison baselines
- Experimental Robust Video Matting MobileNetV3 comparison path with recurrent temporal state, internal downsample, and optional predicted-foreground controls
- Pose Landmarker Lite, Full, and Heavy comparison baselines
- GPU-first local inference with automatic CPU fallback when GPU mask readback is unavailable
- Worker-first inference keeps the segmentation model off the render thread, with a main-thread comparison toggle
- Frame-synchronized video compositing prevents a worker matte from being applied to a newer motion frame
- Blur, four color-first procedural virtual backgrounds (Black, Midnight, Aurora, and Ember), 20 RC backgrounds, and custom image replacement
- Result, segmentation-mask, and split comparison views
- Edge-aware matte refinement, chroma-weighted high-resolution upsampling, hand-safe enclosed-hole filling, separate face/hair recovery and body-skin boundary cleanup, edge-color decontamination, motion-edge cleanup, blur-edge protection, and alpha-preserving background light wrap
- Controls for threshold, feathering, motion-aware temporal smoothing, spatial smoothing, hole filling, edge expansion, face/hair recovery, skin-edge cleanup, edge-color cleanup, multiclass ambiguity suppression, inference rate, model, preprocessing size, delegate, blur renderer, output resolution, and RVM-specific stages
- Live output and matte FPS, inference time, refinement time, render time, mask coverage, mask motion, matte age, memory, and slow-frame statistics
- Fully local model and WebAssembly assets
- Device-local preference persistence for reproducible source, model, mask, effect, performance, and comparison selections
- A live Default/Modified indicator; Reset restores every persisted selection and removes the saved preference record

The image source includes four local benchmark scenes: a neutral portrait, opposing hand poses, and a motion-softened hand frame. Choose `4-up benchmark` to run four independent mattes through the same active model, refinement stages, effect, and background on one 2×2 output canvas.

## Robust Video Matting experiment

The `RVM MobileNetV3` option uses the official quantized TensorFlow.js model from [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/tree/tfjs). It runs in a worker with TensorFlow.js WebGL or CPU execution and preserves four recurrent tensors between video frames. Source changes and video loops reset those tensors.

RVM emits a native soft alpha, so the lab bypasses its MediaPipe-oriented threshold, spatial smoothing, and hole-filling stages for this model. Matte expansion and the blur/replacement compositors remain available. `Predicted foreground` optionally uses RVM's reconstructed color layer; it is off by default because its lower resolution can soften image detail.

The vendored RVM model and code are licensed under GPL-3.0. The upstream license is included at `public/rvm/LICENSE`; review license compatibility before distributing a production build.

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Then open the local URL printed by the development server. Camera input requires browser permission.

## Verification

```bash
npm run lint
npm run typecheck
npm test
```
