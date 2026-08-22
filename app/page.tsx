"use client";

import type {
  ImageSegmenter as ImageSegmenterInstance,
  PoseLandmarker as PoseLandmarkerInstance,
} from "@mediapipe/tasks-vision";
import {
  Activity,
  Aperture,
  Camera,
  CirclePause,
  CirclePlay,
  Cpu,
  Download,
  Gauge,
  Image as ImageIcon,
  Layers3,
  Maximize2,
  MonitorUp,
  RotateCcw,
  ScanLine,
  Settings2,
  Sparkles,
  Upload,
  Video,
  WandSparkles,
} from "lucide-react";
import { ChangeEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

type SourceMode = "image" | "video" | "camera";
type ImageViewMode = "single" | "grid";
type EffectMode = "blur" | "replace" | "none";
type PreviewMode = "result" | "mask" | "split";
type Quality = "fast" | "balanced" | "quality";
type ProcessingDelegate = "GPU" | "CPU";
type DelegatePreference = "auto" | ProcessingDelegate;
type SegmentationModel = "selfie-landscape" | "selfie-multiclass" | "rvm-mobilenetv3" | "pose-lite" | "pose-full" | "pose-heavy";
type BlurRenderMode = "efficient" | "full";
type OutputResolution = "360p" | "540p" | "720p";
type SettingsTab = "effect" | "mask" | "perf";
type SavedSettings = {
  sourceMode: SourceMode;
  imageViewMode: ImageViewMode;
  imagePreset: string;
  videoPreset: string;
  effect: EffectMode;
  previewMode: PreviewMode;
  background: string;
  settingsTab: SettingsTab;
  blurRadius: number;
  backgroundSaturation: number;
  blurEdgeGuard: number;
  lightWrap: number;
  threshold: number;
  feather: number;
  temporalSmoothing: number;
  edgeExpansion: number;
  spatialSmoothing: number;
  detailRecovery: number;
  skinEdgeCleanup: number;
  otherSuppression: number;
  edgeDecontamination: number;
  motionEdgeCleanup: number;
  temporalEnabled: boolean;
  motionAwareTemporal: boolean;
  holeFill: boolean;
  edgeAwareRefinement: boolean;
  guidedUpsampling: number;
  blurRenderMode: BlurRenderMode;
  inferenceRate: number;
  quality: Quality;
  resolution: OutputResolution;
  mirror: boolean;
  adaptiveQuality: boolean;
  workerInference: boolean;
  syncVideoFrames: boolean;
  rvmDownsample: number;
  rvmRecurrent: boolean;
  rvmForeground: boolean;
  comparePosition: number;
  segmentationModel: SegmentationModel;
  delegatePreference: DelegatePreference;
};
type WorkerMaskResult = {
  frameId: number;
  values: ArrayBuffer;
  width: number;
  height: number;
  inferenceMs: number;
  unreadable: boolean;
  foreground?: ImageBitmap;
  benchmarkId?: string;
};
type BenchmarkRawMask = { values: Float32Array; width: number; height: number };
type BenchmarkCompositeMask = { mask: HTMLCanvasElement; blurMask: HTMLCanvasElement; edgeCorrection: HTMLCanvasElement; coverage: number };

type Crop = { x: number; y: number; width: number; height: number };
type ImagePreset = { id: string; name: string; detail: string; src: string; crop: Crop };
type VideoPreset = { id: string; name: string; detail: string; src: string };
type ProceduralBackground = { id: string; name: string; colors: [string, string, string]; accent: string };
type ImageBackground = { id: string; name: string; src: string };
type BackgroundPreset = ProceduralBackground | ImageBackground;

type PipelineStats = {
  fps: number;
  maskFps: number;
  inferenceMs: number;
  refinementMs: number;
  renderMs: number;
  maskCoverage: number;
  maskMotion: number;
  matteAgeMs: number;
  memoryMb: number;
  dropped: number;
};

const SEGMENTATION_MODELS: Record<SegmentationModel, { name: string; detail: string; path: string; task: "selfie" | "pose" | "rvm" }> = {
  "selfie-landscape": {
    name: "Selfie Landscape",
    detail: "Meet-style · 256×144",
    path: "/mediapipe/selfie_segmenter_landscape.tflite",
    task: "selfie",
  },
  "selfie-multiclass": {
    name: "Selfie Multiclass",
    detail: "Hair + skin + clothes",
    path: "/mediapipe/selfie_multiclass_256x256.tflite",
    task: "selfie",
  },
  "rvm-mobilenetv3": {
    name: "RVM MobileNetV3",
    detail: "Experimental · recurrent",
    path: "/rvm/model.json",
    task: "rvm",
  },
  "pose-lite": { name: "Pose Lite", detail: "Comparison · fast", path: "/mediapipe/pose_landmarker_lite.task", task: "pose" },
  "pose-full": { name: "Pose Full", detail: "Comparison · balanced", path: "/mediapipe/pose_landmarker_full.task", task: "pose" },
  "pose-heavy": { name: "Pose Heavy", detail: "Comparison · slow", path: "/mediapipe/pose_landmarker_heavy.task", task: "pose" },
};

const SEGMENTATION_MODEL_ORDER = Object.keys(SEGMENTATION_MODELS) as SegmentationModel[];

const MODEL_INPUT_WIDTHS: Record<Quality, number> = {
  fast: 256,
  balanced: 384,
  quality: 512,
};

const MULTICLASS_INPUT_WIDTHS: Record<Quality, number> = {
  fast: 192,
  balanced: 256,
  quality: 320,
};

const RVM_INPUT_WIDTHS: Record<Quality, number> = {
  fast: 320,
  balanced: 512,
  quality: 640,
};

const getModelInputWidth = (model: SegmentationModel, quality: Quality) =>
  model === "selfie-multiclass"
    ? MULTICLASS_INPUT_WIDTHS[quality]
    : model === "rvm-mobilenetv3"
      ? RVM_INPUT_WIDTHS[quality]
      : MODEL_INPUT_WIDTHS[quality];

const IMAGE_PRESETS: ImagePreset[] = [
  {
    id: "neutral-portrait",
    name: "Neutral portrait",
    detail: "Baseline silhouette",
    src: "/samples/user-photo-1.jpg",
    crop: { x: 0, y: 0, width: 1, height: 1 },
  },
  {
    id: "right-hand",
    name: "Right hand",
    detail: "Finger-edge stress test",
    src: "/samples/user-photo-2.jpg",
    crop: { x: 0, y: 0, width: 1, height: 1 },
  },
  {
    id: "left-hand",
    name: "Left hand",
    detail: "Arm and hand boundary",
    src: "/samples/user-photo-3.jpg",
    crop: { x: 0, y: 0, width: 1, height: 1 },
  },
  {
    id: "motion-still",
    name: "Motion still",
    detail: "Soft hand-edge stress test",
    src: "/samples/user-photo-4.jpg",
    crop: { x: 0, y: 0, width: 1, height: 1 },
  },
];

const VIDEO_PRESETS: VideoPreset[] = [
  { id: "user-motion", name: "Hand motion", detail: "Your movement capture", src: "/samples/user-motion.mov" },
  { id: "pose-motion", name: "Pose motion", detail: "Full-body tracking", src: "/samples/pose-segmentation.mp4" },
];

const IMAGE_BACKGROUNDS: ImageBackground[] = Array.from({ length: 20 }, (_, index) => {
  const number = String(index + 1).padStart(3, "0");
  return { id: `rc-${number}`, name: `RC Studio ${index + 1}`, src: `/backgrounds/00-2026-AI-Gen-${number}.png` };
});

const PROCEDURAL_BACKGROUNDS: ProceduralBackground[] = [
  { id: "black", name: "Black", colors: ["#000000", "#000000", "#000000"], accent: "#000000" },
  { id: "midnight", name: "Midnight studio", colors: ["#07111e", "#12314b", "#0b7782"], accent: "#67e8f9" },
  { id: "aurora", name: "Arctic aurora", colors: ["#07151f", "#1f4f54", "#647a70"], accent: "#a7f3d0" },
  { id: "ember", name: "Warm loft", colors: ["#28120c", "#6b3522", "#bd6d3d"], accent: "#fdba74" },
];

const BACKGROUNDS: BackgroundPreset[] = [...PROCEDURAL_BACKGROUNDS, ...IMAGE_BACKGROUNDS];

const RESOLUTIONS = {
  "360p": { width: 640, height: 360 },
  "540p": { width: 960, height: 540 },
  "720p": { width: 1280, height: 720 },
};

const SETTINGS_STORAGE_KEY = "virtual-background-lab:settings:v1";

const DEFAULT_SETTINGS: SavedSettings = {
  sourceMode: "image",
  imageViewMode: "single",
  imagePreset: IMAGE_PRESETS[0].id,
  videoPreset: VIDEO_PRESETS[0].id,
  effect: "blur",
  previewMode: "result",
  background: BACKGROUNDS[0].id,
  settingsTab: "effect",
  blurRadius: 32,
  backgroundSaturation: 84,
  blurEdgeGuard: 6,
  lightWrap: 18,
  threshold: 50,
  feather: 7,
  temporalSmoothing: 10,
  edgeExpansion: 0,
  spatialSmoothing: 1,
  detailRecovery: 24,
  skinEdgeCleanup: 18,
  otherSuppression: 60,
  edgeDecontamination: 55,
  motionEdgeCleanup: 4,
  temporalEnabled: true,
  motionAwareTemporal: true,
  holeFill: true,
  edgeAwareRefinement: true,
  guidedUpsampling: 65,
  blurRenderMode: "efficient",
  inferenceRate: 30,
  quality: "balanced",
  resolution: "720p",
  mirror: true,
  adaptiveQuality: true,
  workerInference: true,
  syncVideoFrames: true,
  rvmDownsample: 50,
  rvmRecurrent: true,
  rvmForeground: false,
  comparePosition: 50,
  segmentationModel: "selfie-multiclass",
  delegatePreference: "auto",
};

const isOneOf = <Value extends string>(value: string, options: readonly Value[]): value is Value =>
  options.includes(value as Value);

const sanitizeSavedSettings = (input: unknown): SavedSettings => {
  if (!input || typeof input !== "object") return { ...DEFAULT_SETTINGS };
  const candidate = input as Partial<SavedSettings>;
  const settings = { ...DEFAULT_SETTINGS };

  for (const key of Object.keys(DEFAULT_SETTINGS) as (keyof SavedSettings)[]) {
    const value = candidate[key];
    if (typeof value === typeof DEFAULT_SETTINGS[key]) Object.assign(settings, { [key]: value });
  }

  if (!isOneOf(settings.sourceMode, ["image", "video", "camera"] as const)) settings.sourceMode = DEFAULT_SETTINGS.sourceMode;
  if (!isOneOf(settings.imageViewMode, ["single", "grid"] as const)) settings.imageViewMode = DEFAULT_SETTINGS.imageViewMode;
  if (!isOneOf(settings.effect, ["blur", "replace", "none"] as const)) settings.effect = DEFAULT_SETTINGS.effect;
  if (!isOneOf(settings.previewMode, ["result", "mask", "split"] as const)) settings.previewMode = DEFAULT_SETTINGS.previewMode;
  if (!isOneOf(settings.settingsTab, ["effect", "mask", "perf"] as const)) settings.settingsTab = DEFAULT_SETTINGS.settingsTab;
  if (!isOneOf(settings.blurRenderMode, ["efficient", "full"] as const)) settings.blurRenderMode = DEFAULT_SETTINGS.blurRenderMode;
  if (!isOneOf(settings.quality, ["fast", "balanced", "quality"] as const)) settings.quality = DEFAULT_SETTINGS.quality;
  if (!isOneOf(settings.resolution, ["360p", "540p", "720p"] as const)) settings.resolution = DEFAULT_SETTINGS.resolution;
  if (!isOneOf(settings.delegatePreference, ["auto", "GPU", "CPU"] as const)) settings.delegatePreference = DEFAULT_SETTINGS.delegatePreference;
  if (!Object.hasOwn(SEGMENTATION_MODELS, settings.segmentationModel)) settings.segmentationModel = DEFAULT_SETTINGS.segmentationModel;
  if (!IMAGE_PRESETS.some(({ id }) => id === settings.imagePreset)) settings.imagePreset = DEFAULT_SETTINGS.imagePreset;
  if (!VIDEO_PRESETS.some(({ id }) => id === settings.videoPreset)) settings.videoPreset = DEFAULT_SETTINGS.videoPreset;
  if (!BACKGROUNDS.some(({ id }) => id === settings.background)) settings.background = DEFAULT_SETTINGS.background;

  const ranges: Partial<Record<keyof SavedSettings, readonly [number, number]>> = {
    blurRadius: [2, 64],
    backgroundSaturation: [40, 130],
    blurEdgeGuard: [0, 24],
    lightWrap: [0, 30],
    threshold: [10, 85],
    feather: [1, 40],
    temporalSmoothing: [0, 85],
    edgeExpansion: [-3, 4],
    spatialSmoothing: [0, 3],
    detailRecovery: [0, 35],
    skinEdgeCleanup: [0, 50],
    otherSuppression: [0, 80],
    edgeDecontamination: [0, 100],
    motionEdgeCleanup: [0, 10],
    guidedUpsampling: [0, 100],
    inferenceRate: [4, 30],
    rvmDownsample: [25, 100],
    comparePosition: [10, 90],
  };
  for (const [key, range] of Object.entries(ranges) as [keyof SavedSettings, readonly [number, number]][]) {
    const value = settings[key];
    if (typeof value === "number") Object.assign(settings, { [key]: Math.max(range[0], Math.min(range[1], value)) });
  }
  if (settings.segmentationModel === "rvm-mobilenetv3" || settings.imageViewMode === "grid") settings.workerInference = true;
  return settings;
};

const initialStats: PipelineStats = {
  fps: 0,
  maskFps: 0,
  inferenceMs: 0,
  refinementMs: 0,
  renderMs: 0,
  maskCoverage: 0,
  maskMotion: 0,
  matteAgeMs: 0,
  memoryMb: 0,
  dropped: 0,
};

const smoothstep = (start: number, end: number, value: number) => {
  const x = Math.max(0, Math.min(1, (value - start) / Math.max(0.0001, end - start)));
  return x * x * (3 - 2 * x);
};

const getGuidedMatteWidth = (quality: Quality, sourceMode: SourceMode, outputWidth: number) => {
  const staticWidths: Record<Quality, number> = { fast: 320, balanced: 640, quality: 960 };
  const motionWidths: Record<Quality, number> = { fast: 256, balanced: 480, quality: 640 };
  return Math.min(outputWidth, (sourceMode === "image" ? staticWidths : motionWidths)[quality]);
};

function extremaPass(
  source: Float32Array,
  width: number,
  height: number,
  radius: number,
  horizontal: boolean,
  dilate: boolean,
) {
  const output = new Float32Array(source.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = dilate ? 0 : 1;
      for (let offset = -radius; offset <= radius; offset++) {
        const sampleX = horizontal ? Math.max(0, Math.min(width - 1, x + offset)) : x;
        const sampleY = horizontal ? y : Math.max(0, Math.min(height - 1, y + offset));
        const sample = source[sampleY * width + sampleX];
        value = dilate ? Math.max(value, sample) : Math.min(value, sample);
      }
      output[y * width + x] = value;
    }
  }
  return output;
}

function morphMask(source: Float32Array, width: number, height: number, amount: number) {
  if (!amount) return source;
  const radius = Math.abs(Math.round(amount));
  const dilate = amount > 0;
  return extremaPass(extremaPass(source, width, height, radius, true, dilate), width, height, radius, false, dilate);
}

function boxBlurMask(source: Float32Array, width: number, height: number, radius: number) {
  if (!radius) return source;
  const horizontal = new Float32Array(source.length);
  const output = new Float32Array(source.length);
  const size = radius * 2 + 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let total = 0;
      for (let offset = -radius; offset <= radius; offset++) {
        total += source[y * width + Math.max(0, Math.min(width - 1, x + offset))];
      }
      horizontal[y * width + x] = total / size;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let total = 0;
      for (let offset = -radius; offset <= radius; offset++) {
        total += horizontal[Math.max(0, Math.min(height - 1, y + offset)) * width + x];
      }
      output[y * width + x] = total / size;
    }
  }
  return output;
}

function fillEnclosedMaskHoles(source: Float32Array, width: number, height: number) {
  const output = new Float32Array(source);
  const visited = new Uint8Array(source.length);
  const stack = new Int32Array(source.length);
  const maximumHoleArea = Math.max(16, Math.round(source.length * 0.002));

  for (let start = 0; start < source.length; start++) {
    if (visited[start] || source[start] >= 0.45) continue;
    let stackSize = 1;
    let componentSize = 0;
    let touchesBorder = false;
    const component: number[] = [];
    stack[0] = start;
    visited[start] = 1;

    while (stackSize > 0) {
      const index = stack[--stackSize];
      const x = index % width;
      const y = Math.floor(index / width);
      component.push(index);
      componentSize++;
      if (x === 0 || y === 0 || x === width - 1 || y === height - 1) touchesBorder = true;

      const neighbors = [index - 1, index + 1, index - width, index + width];
      for (let direction = 0; direction < neighbors.length; direction++) {
        if ((direction === 0 && x === 0) || (direction === 1 && x === width - 1) || (direction === 2 && y === 0) || (direction === 3 && y === height - 1)) continue;
        const neighbor = neighbors[direction];
        if (!visited[neighbor] && source[neighbor] < 0.45) {
          visited[neighbor] = 1;
          stack[stackSize++] = neighbor;
        }
      }
    }

    if (!touchesBorder && componentSize <= maximumHoleArea) {
      for (const index of component) output[index] = 1;
    }
  }
  return output;
}

function guidedUpsampleMask(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  guide: ImageData,
  strengthPercent: number,
) {
  const targetWidth = guide.width;
  const targetHeight = guide.height;
  const output = new Float32Array(targetWidth * targetHeight);
  const guideData = guide.data;
  const strength = Math.max(0, Math.min(1, strengthPercent / 100));
  const rangeSigma = 54 - strength * 38;
  const rangeDenominator = 2 * rangeSigma * rangeSigma;

  for (let y = 0; y < targetHeight; y++) {
    const sourceY = (y + 0.5) * sourceHeight / targetHeight - 0.5;
    const y0 = Math.max(0, Math.min(sourceHeight - 1, Math.floor(sourceY)));
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const fractionY = Math.max(0, Math.min(1, sourceY - y0));
    for (let x = 0; x < targetWidth; x++) {
      const sourceX = (x + 0.5) * sourceWidth / targetWidth - 0.5;
      const x0 = Math.max(0, Math.min(sourceWidth - 1, Math.floor(sourceX)));
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const fractionX = Math.max(0, Math.min(1, sourceX - x0));
      const guideOffset = (y * targetWidth + x) * 4;
      const red = guideData[guideOffset];
      const green = guideData[guideOffset + 1];
      const blue = guideData[guideOffset + 2];
      let total = 0;
      let weightTotal = 0;

      for (let sampleIndex = 0; sampleIndex < 4; sampleIndex++) {
        const sampleX = sampleIndex % 2 === 0 ? x0 : x1;
        const sampleY = sampleIndex < 2 ? y0 : y1;
        const spatialWeight = (sampleIndex % 2 === 0 ? 1 - fractionX : fractionX)
          * (sampleIndex < 2 ? 1 - fractionY : fractionY);
        const mappedX = Math.max(0, Math.min(targetWidth - 1, Math.round((sampleX + 0.5) * targetWidth / sourceWidth - 0.5)));
        const mappedY = Math.max(0, Math.min(targetHeight - 1, Math.round((sampleY + 0.5) * targetHeight / sourceHeight - 0.5)));
        const mappedOffset = (mappedY * targetWidth + mappedX) * 4;
        const redDelta = red - guideData[mappedOffset];
        const greenDelta = green - guideData[mappedOffset + 1];
        const blueDelta = blue - guideData[mappedOffset + 2];
        const luminanceDelta = redDelta * 0.299 + greenDelta * 0.587 + blueDelta * 0.114;
        const blueChromaDelta = redDelta * -0.169 + greenDelta * -0.331 + blueDelta * 0.5;
        const redChromaDelta = redDelta * 0.5 + greenDelta * -0.419 + blueDelta * -0.081;
        const colorDistance = luminanceDelta * luminanceDelta * 0.35
          + blueChromaDelta * blueChromaDelta * 1.8
          + redChromaDelta * redChromaDelta * 1.8;
        const rangeWeight = Math.exp(-colorDistance / rangeDenominator);
        const weight = spatialWeight * ((1 - strength) + strength * rangeWeight);
        total += source[sampleY * sourceWidth + sampleX] * weight;
        weightTotal += weight;
      }
      output[y * targetWidth + x] = weightTotal > 0.00001 ? total / weightTotal : source[y0 * sourceWidth + x0];
    }
  }
  return output;
}

function edgeAwareSmoothMask(
  source: Float32Array,
  width: number,
  height: number,
  guide: ImageData,
  radius: number,
) {
  if (!radius) return source;
  const softened = boxBlurMask(source, width, height, radius);
  const output = new Float32Array(source.length);
  const guideWidth = guide.width;
  const guideHeight = guide.height;
  const guideData = guide.data;
  const guideScaleX = guideWidth / width;
  const guideScaleY = guideHeight / height;
  const luminance = new Float32Array(source.length);
  for (let y = 0; y < height; y++) {
    const guideY = Math.min(guideHeight - 1, Math.floor(y * guideScaleY));
    for (let x = 0; x < width; x++) {
      const guideX = Math.min(guideWidth - 1, Math.floor(x * guideScaleX));
      const guideOffset = (guideY * guideWidth + guideX) * 4;
      luminance[y * width + x] = (guideData[guideOffset] * 54 + guideData[guideOffset + 1] * 183 + guideData[guideOffset + 2] * 19) / 256;
    }
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const left = luminance[y * width + Math.max(0, x - 1)];
      const right = luminance[y * width + Math.min(width - 1, x + 1)];
      const top = luminance[Math.max(0, y - 1) * width + x];
      const bottom = luminance[Math.min(height - 1, y + 1) * width + x];
      const imageEdge = Math.min(1, (Math.abs(left - right) + Math.abs(top - bottom)) / 52);
      const matteEdge = Math.min(1, Math.abs(source[index] - softened[index]) * 5);
      const preserveDetail = Math.max(imageEdge * 0.82, matteEdge * 0.7);
      output[index] = softened[index] * (1 - preserveDetail) + source[index] * preserveDetail;
    }
  }
  return output;
}

function createEdgeCorrectionCanvas(
  alphaValues: Float32Array,
  width: number,
  height: number,
  foregroundGuide: ImageData,
  replacementGuide: ImageData | undefined,
  cleanupPercent: number,
  lightWrapPercent: number,
) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context || (!cleanupPercent && !lightWrapPercent)) return canvas;
  const correction = context.createImageData(width, height);
  const source = foregroundGuide.data;
  const replacement = replacementGuide?.data;
  const cleanupStrength = cleanupPercent / 100;
  const wrapStrength = lightWrapPercent / 100;
  const searchRadius = 3;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const alpha = alphaValues[index];
      if (alpha <= 0.025 || alpha >= 0.985) continue;
      let interiorIndex = index;
      let interiorAlpha = alpha;
      for (let offsetY = -searchRadius; offsetY <= searchRadius; offsetY++) {
        const sampleY = Math.max(0, Math.min(height - 1, y + offsetY));
        for (let offsetX = -searchRadius; offsetX <= searchRadius; offsetX++) {
          const sampleX = Math.max(0, Math.min(width - 1, x + offsetX));
          const sampleIndex = sampleY * width + sampleX;
          if (alphaValues[sampleIndex] > interiorAlpha) {
            interiorAlpha = alphaValues[sampleIndex];
            interiorIndex = sampleIndex;
          }
        }
      }

      const edgeStrength = 1 - Math.abs(alpha * 2 - 1);
      const cleanupMix = cleanupStrength * edgeStrength * Math.max(0, Math.min(1, (interiorAlpha - alpha) * 1.8));
      const sourceOffset = index * 4;
      const interiorOffset = interiorIndex * 4;
      let red = source[sourceOffset] * (1 - cleanupMix) + source[interiorOffset] * cleanupMix;
      let green = source[sourceOffset + 1] * (1 - cleanupMix) + source[interiorOffset + 1] * cleanupMix;
      let blue = source[sourceOffset + 2] * (1 - cleanupMix) + source[interiorOffset + 2] * cleanupMix;
      let wrapMix = 0;
      if (replacement && wrapStrength > 0) {
        const backgroundLuminance = replacement[sourceOffset] * 0.299
          + replacement[sourceOffset + 1] * 0.587
          + replacement[sourceOffset + 2] * 0.114;
        if (backgroundLuminance > 14) {
          wrapMix = wrapStrength * edgeStrength * 0.35;
          red = red * (1 - wrapMix) + replacement[sourceOffset] * wrapMix;
          green = green * (1 - wrapMix) + replacement[sourceOffset + 1] * wrapMix;
          blue = blue * (1 - wrapMix) + replacement[sourceOffset + 2] * wrapMix;
        }
      }
      const correctionAlpha = Math.max(cleanupMix, wrapMix);
      correction.data[sourceOffset] = Math.round(red);
      correction.data[sourceOffset + 1] = Math.round(green);
      correction.data[sourceOffset + 2] = Math.round(blue);
      correction.data[sourceOffset + 3] = Math.round(correctionAlpha * 255);
    }
  }
  context.putImageData(correction, 0, 0);
  return canvas;
}

function drawCover(
  context: CanvasRenderingContext2D,
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
  crop?: Crop,
) {
  if (crop) {
    context.drawImage(
      source,
      crop.x * sourceWidth,
      crop.y * sourceHeight,
      crop.width * sourceWidth,
      crop.height * sourceHeight,
      0,
      0,
      width,
      height,
    );
    return;
  }

  const scale = Math.max(width / sourceWidth, height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  context.drawImage(source, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
}

function drawPresetBackground(context: CanvasRenderingContext2D, width: number, height: number, preset: ProceduralBackground) {
  const gradient = context.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, preset.colors[0]);
  gradient.addColorStop(0.52, preset.colors[1]);
  gradient.addColorStop(1, preset.colors[2]);
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  const glow = context.createRadialGradient(width * 0.72, height * 0.18, 0, width * 0.72, height * 0.18, width * 0.72);
  glow.addColorStop(0, `${preset.accent}80`);
  glow.addColorStop(0.36, `${preset.accent}24`);
  glow.addColorStop(1, "transparent");
  context.fillStyle = glow;
  context.fillRect(0, 0, width, height);

  context.globalAlpha = 0.16;
  context.strokeStyle = preset.accent;
  context.lineWidth = 1;
  const spacing = Math.max(42, width / 16);
  for (let x = -height; x < width + height; x += spacing) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x + height, height);
    context.stroke();
  }
  context.globalAlpha = 1;
}

function SettingRow({ label, value, children }: { label: string; value?: string; children: ReactNode }) {
  return (
    <div className="space-y-2.5 py-3">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-300">{label}</span>
        {value && <span className="font-mono text-[11px] text-cyan-300">{value}</span>}
      </div>
      {children}
    </div>
  );
}

function Metric({ label, value, unit, tone = "cyan" }: { label: string; value: string; unit?: string; tone?: "cyan" | "lime" | "violet" }) {
  const colors = { cyan: "text-cyan-300", lime: "text-lime-300", violet: "text-violet-300" };
  return (
    <div className="rounded-lg border border-white/[0.07] bg-black/20 p-3">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[.14em] text-slate-600">{label}</div>
      <div className={cn("font-mono text-xl font-semibold tracking-tight", colors[tone])}>
        {value} <span className="text-[10px] font-normal text-slate-600">{unit}</span>
      </div>
    </div>
  );
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const modelCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const blurMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const guideCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const foregroundCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const edgeCorrectionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const backgroundCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const replacementBackgroundCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inferenceWorkerRef = useRef<Worker | undefined>(undefined);
  const workerReadyRef = useRef(false);
  const workerBusyRef = useRef(false);
  const workerResultRef = useRef<WorkerMaskResult | undefined>(undefined);
  const benchmarkImagesRef = useRef<Map<string, HTMLImageElement>>(new Map());
  const benchmarkRawMasksRef = useRef<Map<string, BenchmarkRawMask>>(new Map());
  const benchmarkCompositeMasksRef = useRef<Map<string, BenchmarkCompositeMask>>(new Map());
  const pendingBenchmarkRef = useRef<{ frameId: number; benchmarkId: string } | undefined>(undefined);
  const pendingSyncedFrameRef = useRef<{ id: number; bitmap: ImageBitmap } | undefined>(undefined);
  const syncedFrameRef = useRef<ImageBitmap | undefined>(undefined);
  const rvmForegroundRef = useRef<ImageBitmap | undefined>(undefined);
  const workerFrameIdRef = useRef(0);
  const latestMaskRef = useRef<{ values: Float32Array; width: number; height: number; updatedAt: number } | undefined>(undefined);
  const previousMaskRef = useRef<Float32Array | undefined>(undefined);
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const customBackgroundRef = useRef<HTMLImageElement | undefined>(undefined);
  const presetBackgroundRef = useRef<HTMLImageElement | undefined>(undefined);
  const uploadedSourceUrlRef = useRef<string | undefined>(undefined);
  const customBackgroundUrlRef = useRef<string | undefined>(undefined);
  const frameHistoryRef = useRef<number[]>(Array(36).fill(1));
  const statsAccumulatorRef = useRef({ startedAt: 0, frames: 0, masks: 0, dropped: 0, inferenceMs: 0, refinementMs: 0, renderMs: 0, coverage: 0, motion: 0 });
  const maskSurfaceRef = useRef("Waiting");
  const emptyGpuMaskFramesRef = useRef(0);

  const [sourceMode, setSourceMode] = useState<SourceMode>(DEFAULT_SETTINGS.sourceMode);
  const [imageViewMode, setImageViewMode] = useState<ImageViewMode>(DEFAULT_SETTINGS.imageViewMode);
  const [imagePreset, setImagePreset] = useState(DEFAULT_SETTINGS.imagePreset);
  const [videoPreset, setVideoPreset] = useState(DEFAULT_SETTINGS.videoPreset);
  const [uploadedSource, setUploadedSource] = useState<{ url: string; kind: "image" | "video"; name: string }>();
  const [sourceReady, setSourceReady] = useState(0);
  const [videoPaused, setVideoPaused] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string>();

  const [effect, setEffect] = useState<EffectMode>(DEFAULT_SETTINGS.effect);
  const [previewMode, setPreviewMode] = useState<PreviewMode>(DEFAULT_SETTINGS.previewMode);
  const [background, setBackground] = useState(DEFAULT_SETTINGS.background);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>(DEFAULT_SETTINGS.settingsTab);
  const [customBackgroundUrl, setCustomBackgroundUrl] = useState<string>();
  const [blurRadius, setBlurRadius] = useState(32);
  const [backgroundSaturation, setBackgroundSaturation] = useState(84);
  const [blurEdgeGuard, setBlurEdgeGuard] = useState(DEFAULT_SETTINGS.blurEdgeGuard);
  const [lightWrap, setLightWrap] = useState(18);
  const [threshold, setThreshold] = useState(50);
  const [feather, setFeather] = useState(7);
  const [temporalSmoothing, setTemporalSmoothing] = useState(10);
  const [edgeExpansion, setEdgeExpansion] = useState(0);
  const [spatialSmoothing, setSpatialSmoothing] = useState(1);
  const [detailRecovery, setDetailRecovery] = useState(DEFAULT_SETTINGS.detailRecovery);
  const [skinEdgeCleanup, setSkinEdgeCleanup] = useState(DEFAULT_SETTINGS.skinEdgeCleanup);
  const [otherSuppression, setOtherSuppression] = useState(DEFAULT_SETTINGS.otherSuppression);
  const [edgeDecontamination, setEdgeDecontamination] = useState(DEFAULT_SETTINGS.edgeDecontamination);
  const [motionEdgeCleanup, setMotionEdgeCleanup] = useState(4);
  const [temporalEnabled, setTemporalEnabled] = useState(true);
  const [motionAwareTemporal, setMotionAwareTemporal] = useState(true);
  const [holeFill, setHoleFill] = useState(true);
  const [edgeAwareRefinement, setEdgeAwareRefinement] = useState(true);
  const [guidedUpsampling, setGuidedUpsampling] = useState(DEFAULT_SETTINGS.guidedUpsampling);
  const [blurRenderMode, setBlurRenderMode] = useState<BlurRenderMode>("efficient");
  const [inferenceRate, setInferenceRate] = useState(30);
  const [quality, setQuality] = useState<Quality>("balanced");
  const [resolution, setResolution] = useState<OutputResolution>(DEFAULT_SETTINGS.resolution);
  const [mirror, setMirror] = useState(true);
  const [adaptiveQuality, setAdaptiveQuality] = useState(true);
  const [workerInference, setWorkerInference] = useState(true);
  const [syncVideoFrames, setSyncVideoFrames] = useState(true);
  const [rvmDownsample, setRvmDownsample] = useState(50);
  const [rvmRecurrent, setRvmRecurrent] = useState(true);
  const [rvmForeground, setRvmForeground] = useState(false);
  const [paused, setPaused] = useState(false);
  const [comparePosition, setComparePosition] = useState(50);

  const [modelStatus, setModelStatus] = useState<"loading" | "ready" | "error">("loading");
  const [modelMessage, setModelMessage] = useState("Loading Selfie Multiclass");
  const [landmarker, setLandmarker] = useState<PoseLandmarkerInstance>();
  const [imageSegmenter, setImageSegmenter] = useState<ImageSegmenterInstance>();
  const [segmentationModel, setSegmentationModel] = useState<SegmentationModel>("selfie-multiclass");
  const [delegatePreference, setDelegatePreference] = useState<DelegatePreference>("auto");
  const [processingDelegate, setProcessingDelegate] = useState<ProcessingDelegate>("GPU");
  const [stats, setStats] = useState(initialStats);
  const [frameHistory, setFrameHistory] = useState<number[]>(() => Array(36).fill(1));
  const [maskSurface, setMaskSurface] = useState("Waiting");
  const [settingsHydrated, setSettingsHydrated] = useState(false);

  const applySavedSettings = useCallback((settings: SavedSettings) => {
    setSourceMode(settings.sourceMode);
    setImageViewMode(settings.imageViewMode);
    setImagePreset(settings.imagePreset);
    setVideoPreset(settings.videoPreset);
    setEffect(settings.effect);
    setPreviewMode(settings.previewMode);
    setBackground(settings.background);
    setSettingsTab(settings.settingsTab);
    setBlurRadius(settings.blurRadius);
    setBackgroundSaturation(settings.backgroundSaturation);
    setBlurEdgeGuard(settings.blurEdgeGuard);
    setLightWrap(settings.lightWrap);
    setThreshold(settings.threshold);
    setFeather(settings.feather);
    setTemporalSmoothing(settings.temporalSmoothing);
    setEdgeExpansion(settings.edgeExpansion);
    setSpatialSmoothing(settings.spatialSmoothing);
    setDetailRecovery(settings.detailRecovery);
    setSkinEdgeCleanup(settings.skinEdgeCleanup);
    setOtherSuppression(settings.otherSuppression);
    setEdgeDecontamination(settings.edgeDecontamination);
    setMotionEdgeCleanup(settings.motionEdgeCleanup);
    setTemporalEnabled(settings.temporalEnabled);
    setMotionAwareTemporal(settings.motionAwareTemporal);
    setHoleFill(settings.holeFill);
    setEdgeAwareRefinement(settings.edgeAwareRefinement);
    setGuidedUpsampling(settings.guidedUpsampling);
    setBlurRenderMode(settings.blurRenderMode);
    setInferenceRate(settings.inferenceRate);
    setQuality(settings.quality);
    setResolution(settings.resolution);
    setMirror(settings.mirror);
    setAdaptiveQuality(settings.adaptiveQuality);
    setWorkerInference(settings.workerInference);
    setSyncVideoFrames(settings.syncVideoFrames);
    setRvmDownsample(settings.rvmDownsample);
    setRvmRecurrent(settings.rvmRecurrent);
    setRvmForeground(settings.rvmForeground);
    setComparePosition(settings.comparePosition);
    setSegmentationModel(settings.segmentationModel);
    setDelegatePreference(settings.delegatePreference);
    setProcessingDelegate(settings.delegatePreference === "CPU" ? "CPU" : "GPU");
  }, []);

  const currentSettings = useMemo<SavedSettings>(() => ({
    sourceMode,
    imageViewMode,
    imagePreset,
    videoPreset,
    effect,
    previewMode,
    background,
    settingsTab,
    blurRadius,
    backgroundSaturation,
    blurEdgeGuard,
    lightWrap,
    threshold,
    feather,
    temporalSmoothing,
    edgeExpansion,
    spatialSmoothing,
    detailRecovery,
    skinEdgeCleanup,
    otherSuppression,
    edgeDecontamination,
    motionEdgeCleanup,
    temporalEnabled,
    motionAwareTemporal,
    holeFill,
    edgeAwareRefinement,
    guidedUpsampling,
    blurRenderMode,
    inferenceRate,
    quality,
    resolution,
    mirror,
    adaptiveQuality,
    workerInference,
    syncVideoFrames,
    rvmDownsample,
    rvmRecurrent,
    rvmForeground,
    comparePosition,
    segmentationModel,
    delegatePreference,
  }), [
    adaptiveQuality, background, backgroundSaturation, blurEdgeGuard, blurRadius, blurRenderMode,
    comparePosition, delegatePreference, detailRecovery, edgeAwareRefinement, edgeDecontamination, edgeExpansion, effect,
    feather, guidedUpsampling, holeFill, imagePreset, imageViewMode, inferenceRate, lightWrap, mirror, motionAwareTemporal,
    motionEdgeCleanup, otherSuppression, previewMode, quality, resolution, rvmDownsample, rvmForeground, rvmRecurrent, skinEdgeCleanup,
    segmentationModel, settingsTab, sourceMode, spatialSmoothing, syncVideoFrames, temporalEnabled,
    temporalSmoothing, threshold, videoPreset, workerInference,
  ]);
  const modifiedSettingCount = useMemo(
    () => (Object.keys(DEFAULT_SETTINGS) as (keyof SavedSettings)[])
      .filter((key) => currentSettings[key] !== DEFAULT_SETTINGS[key]).length,
    [currentSettings],
  );

  const selectedImage = useMemo(() => IMAGE_PRESETS.find((item) => item.id === imagePreset) ?? IMAGE_PRESETS[0], [imagePreset]);
  const selectedVideo = useMemo(() => VIDEO_PRESETS.find((item) => item.id === videoPreset) ?? VIDEO_PRESETS[0], [videoPreset]);
  const selectedBackground = useMemo(() => BACKGROUNDS.find((item) => item.id === background) ?? BACKGROUNDS[0], [background]);
  const selectedSegmentationModel = SEGMENTATION_MODELS[segmentationModel];

  const activeSource = uploadedSource?.kind === sourceMode ? uploadedSource.url : sourceMode === "image" ? selectedImage.src : selectedVideo.src;

  useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      try {
        const saved = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
        if (saved) applySavedSettings(sanitizeSavedSettings(JSON.parse(saved)));
      } catch {
        try {
          window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
        } catch {
          // Browser storage can be unavailable in hardened privacy contexts.
        }
      } finally {
        setSettingsHydrated(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [applySavedSettings]);

  useEffect(() => {
    if (!settingsHydrated) return;
    try {
      if (modifiedSettingCount === 0) {
        window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
        return;
      }
      window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(currentSettings));
    } catch {
      // The lab remains usable when storage is blocked or full.
    }
  }, [currentSettings, modifiedSettingCount, settingsHydrated]);

  useEffect(() => {
    if (!settingsHydrated) return;
    let cancelled = false;
    let poseInstance: PoseLandmarkerInstance | undefined;
    let selfieInstance: ImageSegmenterInstance | undefined;
    let inferenceWorker: Worker | undefined;
    const selectedModel = SEGMENTATION_MODELS[segmentationModel];

    void (async () => {
      try {
        setModelStatus("loading");
        setModelMessage(`Loading ${selectedModel.name} · ${processingDelegate}`);
        setLandmarker(undefined);
        setImageSegmenter(undefined);
        workerReadyRef.current = false;
        workerBusyRef.current = false;
        workerResultRef.current = undefined;
        rvmForegroundRef.current?.close();
        rvmForegroundRef.current = undefined;
        pendingSyncedFrameRef.current?.bitmap.close();
        pendingSyncedFrameRef.current = undefined;
        syncedFrameRef.current?.close();
        syncedFrameRef.current = undefined;
        if (workerInference) {
          inferenceWorker = new Worker(new URL("./segmentation.worker.ts", import.meta.url), { type: "module" });
          inferenceWorkerRef.current = inferenceWorker;
          inferenceWorker.onmessage = (event: MessageEvent<{ type: "ready" | "mask" | "error"; message?: string; backend?: string } & Partial<WorkerMaskResult>>) => {
            if (cancelled) return;
            if (event.data.type === "ready") {
              workerReadyRef.current = true;
              setModelStatus("ready");
              const backend = event.data.backend ? ` · ${event.data.backend}` : "";
              setModelMessage(`${selectedModel.name} · worker${backend}`);
            } else if (event.data.type === "mask" && event.data.values && event.data.width && event.data.height) {
              workerResultRef.current?.foreground?.close();
              workerResultRef.current = event.data as WorkerMaskResult;
              workerBusyRef.current = false;
            } else if (event.data.type === "error") {
              workerBusyRef.current = false;
              pendingSyncedFrameRef.current?.bitmap.close();
              pendingSyncedFrameRef.current = undefined;
              if (processingDelegate === "GPU" && delegatePreference === "auto") {
                setModelMessage("Worker GPU unavailable · switching to CPU");
                setProcessingDelegate("CPU");
              } else {
                setModelStatus("error");
                setModelMessage(event.data.message ?? "Worker inference failed");
              }
            }
          };
          inferenceWorker.onerror = () => {
            if (cancelled) return;
            workerBusyRef.current = false;
            setModelStatus("error");
            setModelMessage("Segmentation worker failed");
          };
          inferenceWorker.postMessage({
            type: "init",
            modelPath: selectedModel.path,
            task: selectedModel.task,
            delegate: processingDelegate,
          });
          return;
        }
        const { FilesetResolver, ImageSegmenter, PoseLandmarker } = await import("@mediapipe/tasks-vision");
        const files = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        if (selectedModel.task === "selfie") {
          selfieInstance = await ImageSegmenter.createFromOptions(files, {
            baseOptions: { modelAssetPath: selectedModel.path, delegate: processingDelegate },
            runningMode: "VIDEO",
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });
        } else if (selectedModel.task === "pose") {
          poseInstance = await PoseLandmarker.createFromOptions(files, {
            baseOptions: { modelAssetPath: selectedModel.path, delegate: processingDelegate },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.55,
            outputSegmentationMasks: true,
          });
        } else {
          throw new Error("RVM requires worker inference");
        }
        if (cancelled) {
          poseInstance?.close();
          selfieInstance?.close();
          return;
        }
        latestMaskRef.current = undefined;
        previousMaskRef.current = undefined;
        emptyGpuMaskFramesRef.current = 0;
        maskSurfaceRef.current = "Waiting";
        setMaskSurface("Waiting");
        setLandmarker(poseInstance);
        setImageSegmenter(selfieInstance);
        setModelStatus("ready");
        setModelMessage(`${selectedModel.name} · ${processingDelegate}`);
      } catch (error) {
        if (cancelled) return;
        if (processingDelegate === "GPU" && delegatePreference === "auto") {
          setModelMessage("GPU unavailable · switching to CPU");
          setProcessingDelegate("CPU");
          return;
        }
        setModelStatus("error");
        setModelMessage(error instanceof Error ? error.message : "Model could not initialize");
      }
    })();

    return () => {
      cancelled = true;
      inferenceWorker?.terminate();
      if (inferenceWorkerRef.current === inferenceWorker) inferenceWorkerRef.current = undefined;
      poseInstance?.close();
      selfieInstance?.close();
    };
  }, [delegatePreference, processingDelegate, segmentationModel, settingsHydrated, workerInference]);

  useEffect(() => {
    if (!customBackgroundUrl) {
      customBackgroundRef.current = undefined;
      return;
    }
    const image = new Image();
    image.onload = () => {
      customBackgroundRef.current = image;
      setSourceReady((value) => value + 1);
    };
    image.src = customBackgroundUrl;
  }, [customBackgroundUrl]);

  useEffect(() => {
    presetBackgroundRef.current = undefined;
    if (!("src" in selectedBackground)) return;
    const image = new Image();
    image.onload = () => {
      presetBackgroundRef.current = image;
      setSourceReady((value) => value + 1);
    };
    image.src = selectedBackground.src;
    return () => {
      image.onload = null;
    };
  }, [selectedBackground]);

  useEffect(() => {
    let cancelled = false;
    let remaining = IMAGE_PRESETS.length;
    const images = new Map<string, HTMLImageElement>();
    const markReady = () => {
      remaining--;
      if (!cancelled && remaining === 0) setSourceReady((value) => value + 1);
    };
    for (const preset of IMAGE_PRESETS) {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = markReady;
      image.onerror = markReady;
      image.src = preset.src;
      images.set(preset.id, image);
    }
    benchmarkImagesRef.current = images;
    return () => {
      cancelled = true;
      for (const image of images.values()) {
        image.onload = null;
        image.onerror = null;
      }
    };
  }, []);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    pendingSyncedFrameRef.current?.bitmap.close();
    syncedFrameRef.current?.close();
    rvmForegroundRef.current?.close();
    if (uploadedSourceUrlRef.current) URL.revokeObjectURL(uploadedSourceUrlRef.current);
    if (customBackgroundUrlRef.current) URL.revokeObjectURL(customBackgroundUrlRef.current);
  }, []);

  const clearMaskHistory = useCallback(() => {
    latestMaskRef.current = undefined;
    previousMaskRef.current = undefined;
    benchmarkRawMasksRef.current.clear();
    benchmarkCompositeMasksRef.current.clear();
    pendingBenchmarkRef.current = undefined;
    pendingSyncedFrameRef.current?.bitmap.close();
    pendingSyncedFrameRef.current = undefined;
    syncedFrameRef.current?.close();
    syncedFrameRef.current = undefined;
    rvmForegroundRef.current?.close();
    rvmForegroundRef.current = undefined;
    inferenceWorkerRef.current?.postMessage({ type: "reset" });
    workerFrameIdRef.current++;
    statsAccumulatorRef.current.coverage = 0;
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = undefined;
    if (videoRef.current) videoRef.current.srcObject = null;
    clearMaskHistory();
    setCameraActive(false);
  }, [clearMaskHistory]);

  const startCamera = useCallback(async () => {
    try {
      stopCamera();
      setCameraError(undefined);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setSourceReady((value) => value + 1);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "Camera permission was denied");
    }
  }, [stopCamera]);

  const selectSourceMode = (value: string) => {
    const nextMode = value as SourceMode;
    if (nextMode !== "camera") stopCamera();
    clearMaskHistory();
    setSourceMode(nextMode);
  };

  const selectImageViewMode = (value: ImageViewMode) => {
    clearUploadedSource();
    clearMaskHistory();
    setSourceMode("image");
    setImageViewMode(value);
    if (value === "grid") setWorkerInference(true);
  };

  const toggleVideoPlayback = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      try {
        await video.play();
        setVideoPaused(false);
      } catch {
        setVideoPaused(true);
      }
    } else {
      video.pause();
      setVideoPaused(true);
    }
  };

  const clearUploadedSource = () => {
    if (uploadedSourceUrlRef.current) URL.revokeObjectURL(uploadedSourceUrlRef.current);
    uploadedSourceUrlRef.current = undefined;
    setUploadedSource(undefined);
  };

  const clearCustomBackground = () => {
    if (customBackgroundUrlRef.current) URL.revokeObjectURL(customBackgroundUrlRef.current);
    customBackgroundUrlRef.current = undefined;
    setCustomBackgroundUrl(undefined);
  };

  const handleSourceUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    clearUploadedSource();
    const kind = file.type.startsWith("video/") ? "video" : "image";
    const url = URL.createObjectURL(file);
    uploadedSourceUrlRef.current = url;
    clearMaskHistory();
    setUploadedSource({ url, kind, name: file.name });
    setSourceMode(kind);
    if (kind === "image") setImageViewMode("single");
    setSourceReady((value) => value + 1);
  };

  const handleBackgroundUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    clearCustomBackground();
    const url = URL.createObjectURL(file);
    customBackgroundUrlRef.current = url;
    setCustomBackgroundUrl(url);
    setEffect("replace");
  };

  const selectSegmentationModel = (value: SegmentationModel) => {
    clearMaskHistory();
    if (value === "rvm-mobilenetv3") setWorkerInference(true);
    setSegmentationModel(value);
    if (delegatePreference === "auto") setProcessingDelegate("GPU");
  };

  const selectDelegatePreference = (value: DelegatePreference) => {
    clearMaskHistory();
    setDelegatePreference(value);
    setProcessingDelegate(value === "CPU" ? "CPU" : "GPU");
  };

  useEffect(() => {
    const output = canvasRef.current;
    const image = imageRef.current;
    const video = videoRef.current;
    if (!output || paused || !settingsHydrated) return;

    const { width, height } = RESOLUTIONS[resolution];
    output.width = width;
    output.height = height;
    const outputContext = output.getContext("2d", { alpha: false });
    if (!outputContext) return;

    const sourceCanvas = sourceCanvasRef.current ?? document.createElement("canvas");
    const modelCanvas = modelCanvasRef.current ?? document.createElement("canvas");
    const maskCanvas = maskCanvasRef.current ?? document.createElement("canvas");
    const blurMaskCanvas = blurMaskCanvasRef.current ?? document.createElement("canvas");
    const guideCanvas = guideCanvasRef.current ?? document.createElement("canvas");
    const foregroundCanvas = foregroundCanvasRef.current ?? document.createElement("canvas");
    const edgeCorrectionCanvas = edgeCorrectionCanvasRef.current ?? document.createElement("canvas");
    const backgroundCanvas = backgroundCanvasRef.current ?? document.createElement("canvas");
    const replacementBackgroundCanvas = replacementBackgroundCanvasRef.current ?? document.createElement("canvas");
    sourceCanvasRef.current = sourceCanvas;
    modelCanvasRef.current = modelCanvas;
    maskCanvasRef.current = maskCanvas;
    blurMaskCanvasRef.current = blurMaskCanvas;
    guideCanvasRef.current = guideCanvas;
    foregroundCanvasRef.current = foregroundCanvas;
    edgeCorrectionCanvasRef.current = edgeCorrectionCanvas;
    backgroundCanvasRef.current = backgroundCanvas;
    replacementBackgroundCanvasRef.current = replacementBackgroundCanvas;
    sourceCanvas.width = width;
    sourceCanvas.height = height;
    foregroundCanvas.width = width;
    foregroundCanvas.height = height;
    edgeCorrectionCanvas.width = 1;
    edgeCorrectionCanvas.height = 1;
    replacementBackgroundCanvas.width = width;
    replacementBackgroundCanvas.height = height;
    const modelWidth = getModelInputWidth(segmentationModel, quality);
    const modelHeight = Math.round((modelWidth * 9) / 16);
    modelCanvas.width = modelWidth;
    modelCanvas.height = modelHeight;
    // MediaPipe owns the segmentation-mask dimensions. They are not
    // guaranteed to match the canvas supplied to the inference task.
    maskCanvas.width = 1;
    maskCanvas.height = 1;
    blurMaskCanvas.width = 1;
    blurMaskCanvas.height = 1;
    benchmarkCompositeMasksRef.current.clear();

    const sourceContext = sourceCanvas.getContext("2d", { alpha: false });
    const modelContext = modelCanvas.getContext("2d", { alpha: false });
    const maskContext = maskCanvas.getContext("2d");
    const blurMaskContext = blurMaskCanvas.getContext("2d");
    const guideContext = guideCanvas.getContext("2d", { alpha: false });
    const foregroundContext = foregroundCanvas.getContext("2d");
    const edgeCorrectionContext = edgeCorrectionCanvas.getContext("2d");
    const backgroundContext = backgroundCanvas.getContext("2d", { alpha: false });
    const replacementBackgroundContext = replacementBackgroundCanvas.getContext("2d", { alpha: false });
    if (!sourceContext || !modelContext || !maskContext || !blurMaskContext || !guideContext || !foregroundContext || !edgeCorrectionContext || !backgroundContext || !replacementBackgroundContext) return;

    replacementBackgroundContext.fillStyle = "#05080d";
    replacementBackgroundContext.fillRect(0, 0, width, height);
    if (customBackgroundRef.current) {
      drawCover(
        replacementBackgroundContext,
        customBackgroundRef.current,
        customBackgroundRef.current.naturalWidth,
        customBackgroundRef.current.naturalHeight,
        width,
        height,
      );
    } else if (presetBackgroundRef.current) {
      drawCover(
        replacementBackgroundContext,
        presetBackgroundRef.current,
        presetBackgroundRef.current.naturalWidth,
        presetBackgroundRef.current.naturalHeight,
        width,
        height,
      );
    } else if ("colors" in selectedBackground) {
      drawPresetBackground(replacementBackgroundContext, width, height, selectedBackground);
    }

    let animationFrame = 0;
    let lastInferenceAt = -Infinity;
    let lastTimestamp = 0;
    let lastProcessedMaskAt = -1;
    let lastVideoTime = -1;

    const clearInvalidMask = () => {
      latestMaskRef.current = undefined;
      previousMaskRef.current = undefined;
      statsAccumulatorRef.current.coverage = 0;
    };

    const acceptMask = (values: Float32Array, maskWidth: number, maskHeight: number, unreadableGpuMask = false) => {
      if (maskWidth <= 0 || maskHeight <= 0 || values.length !== maskWidth * maskHeight) {
        clearInvalidMask();
        return;
      }

      if (processingDelegate === "GPU" && unreadableGpuMask) {
        emptyGpuMaskFramesRef.current++;
        if (emptyGpuMaskFramesRef.current >= 3) {
          clearInvalidMask();
          if (delegatePreference === "auto") {
            setModelStatus("loading");
            setModelMessage("GPU mask unavailable · switching to CPU");
            setProcessingDelegate("CPU");
          } else {
            setModelMessage("GPU mask readback is empty");
          }
        }
        return;
      }

      emptyGpuMaskFramesRef.current = 0;
      latestMaskRef.current = { values, width: maskWidth, height: maskHeight, updatedAt: performance.now() };
      statsAccumulatorRef.current.masks++;
      const nextMaskSurface = `${maskWidth}×${maskHeight}`;
      if (maskSurfaceRef.current !== nextMaskSurface) {
        maskSurfaceRef.current = nextMaskSurface;
        setMaskSurface(nextMaskSurface);
      }
    };

    const tileWidth = Math.floor(width / 2);
    const tileHeight = Math.floor(height / 2);
    const tileSourceCanvas = document.createElement("canvas");
    const tileOutputCanvas = document.createElement("canvas");
    const tileForegroundCanvas = document.createElement("canvas");
    const tileBlurCanvas = document.createElement("canvas");
    tileSourceCanvas.width = tileOutputCanvas.width = tileForegroundCanvas.width = tileWidth;
    tileSourceCanvas.height = tileOutputCanvas.height = tileForegroundCanvas.height = tileHeight;
    const tileSourceContext = tileSourceCanvas.getContext("2d", { alpha: false });
    const tileOutputContext = tileOutputCanvas.getContext("2d", { alpha: false });
    const tileForegroundContext = tileForegroundCanvas.getContext("2d");
    const tileBlurContext = tileBlurCanvas.getContext("2d", { alpha: false });

    const buildBenchmarkComposite = (
      preset: ImagePreset,
      benchmarkImage: HTMLImageElement,
      rawMask: BenchmarkRawMask,
    ): BenchmarkCompositeMask | undefined => {
      if (!tileSourceContext) return undefined;
      const refinementStartedAt = performance.now();
      modelContext.clearRect(0, 0, modelWidth, modelHeight);
      drawCover(modelContext, benchmarkImage, benchmarkImage.naturalWidth, benchmarkImage.naturalHeight, modelWidth, modelHeight, preset.crop);

      const isRvm = segmentationModel === "rvm-mobilenetv3";
      let refinedMask: Float32Array<ArrayBufferLike> = new Float32Array(rawMask.values);
      if (!isRvm && spatialSmoothing > 0) {
        refinedMask = edgeAwareRefinement
          ? edgeAwareSmoothMask(
              refinedMask,
              rawMask.width,
              rawMask.height,
              modelContext.getImageData(0, 0, modelWidth, modelHeight),
              spatialSmoothing,
            )
          : boxBlurMask(refinedMask, rawMask.width, rawMask.height, spatialSmoothing);
      }
      if (!isRvm && holeFill) refinedMask = fillEnclosedMaskHoles(refinedMask, rawMask.width, rawMask.height);
      if (edgeExpansion) refinedMask = morphMask(refinedMask, rawMask.width, rawMask.height, edgeExpansion);

      const useGuidedUpsampling = !isRvm && guidedUpsampling > 0;
      const compositeWidth = useGuidedUpsampling ? getGuidedMatteWidth(quality, "image", tileWidth) : rawMask.width;
      const compositeHeight = useGuidedUpsampling ? Math.round(compositeWidth * tileHeight / tileWidth) : rawMask.height;
      let compositeMask = refinedMask;
      if (useGuidedUpsampling) {
        guideCanvas.width = compositeWidth;
        guideCanvas.height = compositeHeight;
        guideContext.clearRect(0, 0, compositeWidth, compositeHeight);
        drawCover(guideContext, benchmarkImage, benchmarkImage.naturalWidth, benchmarkImage.naturalHeight, compositeWidth, compositeHeight, preset.crop);
        compositeMask = guidedUpsampleMask(
          refinedMask,
          rawMask.width,
          rawMask.height,
          guideContext.getImageData(0, 0, compositeWidth, compositeHeight),
          guidedUpsampling,
        );
      } else {
        guideCanvas.width = compositeWidth;
        guideCanvas.height = compositeHeight;
        guideContext.clearRect(0, 0, compositeWidth, compositeHeight);
        drawCover(guideContext, benchmarkImage, benchmarkImage.naturalWidth, benchmarkImage.naturalHeight, compositeWidth, compositeHeight, preset.crop);
      }

      const matte = document.createElement("canvas");
      const guardedMatte = document.createElement("canvas");
      matte.width = guardedMatte.width = compositeWidth;
      matte.height = guardedMatte.height = compositeHeight;
      const matteContext = matte.getContext("2d");
      const guardedContext = guardedMatte.getContext("2d");
      if (!matteContext || !guardedContext) return undefined;
      const imageData = matteContext.createImageData(compositeWidth, compositeHeight);
      const guardedImageData = guardedContext.createImageData(compositeWidth, compositeHeight);
      const alphaValues = new Float32Array(compositeMask.length);
      const thresholdValue = Math.max(0.05, threshold / 100);
      const featherValue = Math.max(0.015, feather / 100);
      let foregroundPixels = 0;
      for (let index = 0; index < compositeMask.length; index++) {
        const alpha = isRvm
          ? Math.max(0, Math.min(1, compositeMask[index]))
          : smoothstep(thresholdValue - featherValue, thresholdValue + featherValue, compositeMask[index]);
        const guardedAlpha = Math.max(
          alpha,
          smoothstep(thresholdValue - featherValue - blurEdgeGuard / 100, thresholdValue + featherValue, compositeMask[index]),
        );
        alphaValues[index] = alpha;
        if (alpha >= 0.5) foregroundPixels++;
        const offset = index * 4;
        imageData.data[offset] = guardedImageData.data[offset] = 255;
        imageData.data[offset + 1] = guardedImageData.data[offset + 1] = 255;
        imageData.data[offset + 2] = guardedImageData.data[offset + 2] = 255;
        imageData.data[offset + 3] = Math.round(alpha * 255);
        guardedImageData.data[offset + 3] = Math.round(guardedAlpha * 255);
      }
      matteContext.putImageData(imageData, 0, 0);
      guardedContext.putImageData(guardedImageData, 0, 0);
      const replacementGuideCanvas = document.createElement("canvas");
      replacementGuideCanvas.width = compositeWidth;
      replacementGuideCanvas.height = compositeHeight;
      const replacementGuideContext = replacementGuideCanvas.getContext("2d", { alpha: false });
      replacementGuideContext?.drawImage(replacementBackgroundCanvas, 0, 0, compositeWidth, compositeHeight);
      const edgeCorrection = createEdgeCorrectionCanvas(
        alphaValues,
        compositeWidth,
        compositeHeight,
        guideContext.getImageData(0, 0, compositeWidth, compositeHeight),
        effect === "replace" && replacementGuideContext
          ? replacementGuideContext.getImageData(0, 0, compositeWidth, compositeHeight)
          : undefined,
        edgeDecontamination,
        effect === "replace" ? lightWrap : 0,
      );
      statsAccumulatorRef.current.refinementMs = performance.now() - refinementStartedAt;
      return { mask: matte, blurMask: guardedMatte, edgeCorrection, coverage: foregroundPixels / compositeMask.length * 100 };
    };

    const updateStats = (now: number, renderMs: number, matteAgeMs: number) => {
      const accumulator = statsAccumulatorRef.current;
      accumulator.frames++;
      accumulator.renderMs = renderMs;
      if (renderMs > 33.3) accumulator.dropped++;
      const elapsed = now - accumulator.startedAt;
      if (elapsed < 600) return;
      const fps = accumulator.frames * 1000 / elapsed;
      const memory = performance as Performance & { memory?: { usedJSHeapSize: number } };
      setStats({
        fps,
        maskFps: accumulator.masks * 1000 / elapsed,
        inferenceMs: accumulator.inferenceMs,
        refinementMs: accumulator.refinementMs,
        renderMs: accumulator.renderMs,
        maskCoverage: accumulator.coverage,
        maskMotion: accumulator.motion,
        matteAgeMs,
        memoryMb: memory.memory ? memory.memory.usedJSHeapSize / 1024 / 1024 : 0,
        dropped: accumulator.dropped,
      });
      frameHistoryRef.current = [...frameHistoryRef.current.slice(1), Math.min(1, fps / 30)];
      setFrameHistory(frameHistoryRef.current);
      statsAccumulatorRef.current = { ...accumulator, startedAt: now, frames: 0, masks: 0, dropped: 0 };
    };

    const renderBenchmarkGrid = (now: number) => {
      if (!tileSourceContext || !tileOutputContext || !tileForegroundContext || !tileBlurContext) return;
      const frameStartedAt = performance.now();
      const workerResult = workerResultRef.current;
      if (workerResult) {
        workerResultRef.current = undefined;
        const pending = pendingBenchmarkRef.current;
        if (workerResult.benchmarkId && pending?.frameId === workerResult.frameId && pending.benchmarkId === workerResult.benchmarkId) {
          pendingBenchmarkRef.current = undefined;
          workerResult.foreground?.close();
          statsAccumulatorRef.current.inferenceMs = workerResult.inferenceMs;
          if (!workerResult.unreadable) {
            benchmarkRawMasksRef.current.set(workerResult.benchmarkId, {
              values: new Float32Array(workerResult.values),
              width: workerResult.width,
              height: workerResult.height,
            });
            statsAccumulatorRef.current.masks++;
            const nextMaskSurface = `4 × ${workerResult.width}×${workerResult.height}`;
            if (maskSurfaceRef.current !== nextMaskSurface) {
              maskSurfaceRef.current = nextMaskSurface;
              setMaskSurface(nextMaskSurface);
            }
          }
        } else {
          workerResult.foreground?.close();
        }
      }

      for (const preset of IMAGE_PRESETS) {
        const rawMask = benchmarkRawMasksRef.current.get(preset.id);
        const benchmarkImage = benchmarkImagesRef.current.get(preset.id);
        if (rawMask && benchmarkImage?.complete && !benchmarkCompositeMasksRef.current.has(preset.id)) {
          const composite = buildBenchmarkComposite(preset, benchmarkImage, rawMask);
          if (composite) benchmarkCompositeMasksRef.current.set(preset.id, composite);
        }
      }

      if (workerReadyRef.current && !workerBusyRef.current && !pendingBenchmarkRef.current && inferenceWorkerRef.current) {
        const nextPreset = IMAGE_PRESETS.find((preset) => {
          const benchmarkImage = benchmarkImagesRef.current.get(preset.id);
          return benchmarkImage?.complete && benchmarkImage.naturalWidth > 0 && !benchmarkRawMasksRef.current.has(preset.id);
        });
        const benchmarkImage = nextPreset ? benchmarkImagesRef.current.get(nextPreset.id) : undefined;
        if (nextPreset && benchmarkImage) {
          modelContext.clearRect(0, 0, modelWidth, modelHeight);
          drawCover(modelContext, benchmarkImage, benchmarkImage.naturalWidth, benchmarkImage.naturalHeight, modelWidth, modelHeight, nextPreset.crop);
          workerBusyRef.current = true;
          const activeWorker = inferenceWorkerRef.current;
          const frameId = ++workerFrameIdRef.current;
          pendingBenchmarkRef.current = { frameId, benchmarkId: nextPreset.id };
          if (segmentationModel === "rvm-mobilenetv3") activeWorker.postMessage({ type: "reset" });
          void createImageBitmap(modelCanvas).then((bitmap) => {
            if (inferenceWorkerRef.current !== activeWorker || pendingBenchmarkRef.current?.frameId !== frameId) {
              bitmap.close();
              workerBusyRef.current = false;
              return;
            }
            activeWorker.postMessage({
              type: "frame",
              frameId,
              benchmarkId: nextPreset.id,
              bitmap,
              timestamp: Math.max(now, frameId),
              detailRecovery,
              skinEdgeCleanup,
              otherSuppression,
              rvmDownsample: rvmDownsample / 100,
              rvmRecurrent: false,
              rvmForeground: false,
            }, [bitmap]);
          }).catch((error: unknown) => {
            workerBusyRef.current = false;
            pendingBenchmarkRef.current = undefined;
            setModelMessage(error instanceof Error ? error.message : "Could not transfer benchmark frame");
          });
        }
      }

      outputContext.fillStyle = "#03060a";
      outputContext.fillRect(0, 0, width, height);
      let coverageTotal = 0;
      let completed = 0;
      IMAGE_PRESETS.forEach((preset, index) => {
        const benchmarkImage = benchmarkImagesRef.current.get(preset.id);
        const composite = benchmarkCompositeMasksRef.current.get(preset.id);
        const column = index % 2;
        const row = Math.floor(index / 2);
        const tileX = column * tileWidth;
        const tileY = row * tileHeight;
        tileSourceContext.clearRect(0, 0, tileWidth, tileHeight);
        if (benchmarkImage?.complete && benchmarkImage.naturalWidth > 0) {
          drawCover(tileSourceContext, benchmarkImage, benchmarkImage.naturalWidth, benchmarkImage.naturalHeight, tileWidth, tileHeight, preset.crop);
        }
        tileOutputContext.clearRect(0, 0, tileWidth, tileHeight);
        if (!composite && effect !== "none") {
          tileOutputContext.drawImage(tileSourceCanvas, 0, 0);
          tileOutputContext.fillStyle = "rgba(2,6,12,.42)";
          tileOutputContext.fillRect(0, 0, tileWidth, tileHeight);
        } else if (effect === "none") {
          tileOutputContext.drawImage(tileSourceCanvas, 0, 0);
        } else if (effect === "blur") {
          if (blurRenderMode === "efficient") {
            const scale = 0.25;
            const blurWidth = Math.max(1, Math.round(tileWidth * scale));
            const blurHeight = Math.max(1, Math.round(tileHeight * scale));
            tileBlurCanvas.width = blurWidth;
            tileBlurCanvas.height = blurHeight;
            const scaledRadius = Math.max(2, blurRadius * scale);
            const overscan = Math.ceil(scaledRadius * 1.8);
            tileBlurContext.filter = `blur(${scaledRadius}px) saturate(${backgroundSaturation}%)`;
            tileBlurContext.drawImage(tileSourceCanvas, -overscan, -overscan, blurWidth + overscan * 2, blurHeight + overscan * 2);
            tileBlurContext.filter = "none";
            tileOutputContext.imageSmoothingEnabled = true;
            tileOutputContext.imageSmoothingQuality = "high";
            tileOutputContext.drawImage(tileBlurCanvas, 0, 0, tileWidth, tileHeight);
          } else {
            const overscan = Math.ceil(blurRadius * 1.8);
            tileOutputContext.filter = `blur(${blurRadius}px) saturate(${backgroundSaturation}%)`;
            tileOutputContext.drawImage(tileSourceCanvas, -overscan, -overscan, tileWidth + overscan * 2, tileHeight + overscan * 2);
            tileOutputContext.filter = "none";
          }
        } else {
          tileOutputContext.drawImage(replacementBackgroundCanvas, 0, 0, width, height, 0, 0, tileWidth, tileHeight);
        }

        if (composite && effect !== "none") {
          completed++;
          coverageTotal += composite.coverage;
          tileForegroundContext.clearRect(0, 0, tileWidth, tileHeight);
          tileForegroundContext.globalCompositeOperation = "source-over";
          tileForegroundContext.drawImage(tileSourceCanvas, 0, 0);
          tileForegroundContext.globalCompositeOperation = "destination-in";
          tileForegroundContext.drawImage(effect === "blur" && blurEdgeGuard > 0 ? composite.blurMask : composite.mask, 0, 0, tileWidth, tileHeight);
          tileForegroundContext.globalCompositeOperation = "source-atop";
          tileForegroundContext.drawImage(composite.edgeCorrection, 0, 0, tileWidth, tileHeight);
          tileForegroundContext.globalCompositeOperation = "source-over";
          tileOutputContext.drawImage(tileForegroundCanvas, 0, 0);
        }
        if (previewMode === "mask") {
          tileOutputContext.fillStyle = "#03060a";
          tileOutputContext.fillRect(0, 0, tileWidth, tileHeight);
          if (composite) tileOutputContext.drawImage(composite.mask, 0, 0, tileWidth, tileHeight);
        } else if (previewMode === "split" && composite) {
          const splitX = Math.round(comparePosition / 100 * tileWidth);
          tileOutputContext.save();
          tileOutputContext.beginPath();
          tileOutputContext.rect(splitX, 0, tileWidth - splitX, tileHeight);
          tileOutputContext.clip();
          tileOutputContext.drawImage(tileSourceCanvas, 0, 0);
          tileOutputContext.restore();
          tileOutputContext.fillStyle = "rgba(255,255,255,.9)";
          tileOutputContext.fillRect(splitX - 1, 0, 2, tileHeight);
        }
        outputContext.drawImage(tileOutputCanvas, tileX, tileY);
        outputContext.fillStyle = "rgba(3,6,10,.72)";
        outputContext.fillRect(tileX + 10, tileY + 10, Math.min(178, tileWidth - 20), 28);
        outputContext.fillStyle = "#e2e8f0";
        outputContext.font = "600 13px ui-sans-serif, system-ui, sans-serif";
        outputContext.fillText(preset.name, tileX + 20, tileY + 29);
        if (!composite && effect !== "none") {
          outputContext.fillStyle = "rgba(226,232,240,.82)";
          outputContext.font = "500 12px ui-monospace, monospace";
          outputContext.textAlign = "center";
          outputContext.fillText("Preparing matte…", tileX + tileWidth / 2, tileY + tileHeight / 2);
          outputContext.textAlign = "start";
        }
      });
      outputContext.fillStyle = "rgba(255,255,255,.28)";
      outputContext.fillRect(tileWidth - 0.5, 0, 1, height);
      outputContext.fillRect(0, tileHeight - 0.5, width, 1);
      statsAccumulatorRef.current.coverage = completed ? coverageTotal / completed : 0;
      statsAccumulatorRef.current.motion = 0;
      updateStats(now, performance.now() - frameStartedAt, 0);
    };

    const render = (now: number) => {
      if (sourceMode === "image" && imageViewMode === "grid" && !uploadedSource) {
        renderBenchmarkGrid(now);
        animationFrame = requestAnimationFrame(render);
        return;
      }
      const sourceElement = sourceMode === "image" ? image : video;
      const sourceWidth = sourceMode === "image" ? image?.naturalWidth ?? 0 : video?.videoWidth ?? 0;
      const sourceHeight = sourceMode === "image" ? image?.naturalHeight ?? 0 : video?.videoHeight ?? 0;

      if (!sourceElement || !sourceWidth || !sourceHeight) {
        animationFrame = requestAnimationFrame(render);
        return;
      }

      if (sourceMode === "video" && video) {
        if (lastVideoTime >= 0 && video.currentTime + 0.05 < lastVideoTime) {
          clearMaskHistory();
          lastProcessedMaskAt = -1;
        }
        lastVideoTime = video.currentTime;
      }

      const frameStartedAt = performance.now();
      sourceContext.save();
      sourceContext.clearRect(0, 0, width, height);
      if (mirror && sourceMode === "camera") {
        sourceContext.translate(width, 0);
        sourceContext.scale(-1, 1);
      }
      drawCover(sourceContext, sourceElement, sourceWidth, sourceHeight, width, height, sourceMode === "image" && !uploadedSource ? selectedImage.crop : undefined);
      sourceContext.restore();

      modelContext.clearRect(0, 0, modelWidth, modelHeight);
      modelContext.drawImage(sourceCanvas, 0, 0, modelWidth, modelHeight);

      const workerResult = workerResultRef.current;
      if (workerResult) {
        workerResultRef.current = undefined;
        const pendingFrame = pendingSyncedFrameRef.current;
        const expectsSyncedFrame = workerInference && syncVideoFrames && sourceMode === "video";
        const isCurrentResult = workerResult.frameId === workerFrameIdRef.current;
        if (isCurrentResult && (!expectsSyncedFrame || pendingFrame?.id === workerResult.frameId)) {
          if (expectsSyncedFrame && pendingFrame) {
            syncedFrameRef.current?.close();
            syncedFrameRef.current = pendingFrame.bitmap;
            pendingSyncedFrameRef.current = undefined;
          }
          rvmForegroundRef.current?.close();
          rvmForegroundRef.current = workerResult.foreground;
          statsAccumulatorRef.current.inferenceMs = workerResult.inferenceMs;
          acceptMask(
            new Float32Array(workerResult.values),
            workerResult.width,
            workerResult.height,
            workerResult.unreadable,
          );
        } else {
          workerResult.foreground?.close();
        }
      }

      const requestedInterval = 1000 / inferenceRate;
      const targetInterval = adaptiveQuality
        ? Math.max(requestedInterval, statsAccumulatorRef.current.inferenceMs * (workerInference ? 1.05 : 1.35))
        : requestedInterval;
      const needsInference = sourceMode !== "image" || !latestMaskRef.current;
      const inferenceReady = workerInference ? workerReadyRef.current && !workerBusyRef.current : Boolean(landmarker || imageSegmenter);
      if (needsInference && inferenceReady && now - lastInferenceAt >= targetInterval) {
        lastInferenceAt = now;
        const inferenceStartedAt = performance.now();
        const timestamp = Math.max(lastTimestamp + 1, now);
        lastTimestamp = timestamp;
        try {
          if (workerInference && inferenceWorkerRef.current) {
            workerBusyRef.current = true;
            const activeWorker = inferenceWorkerRef.current;
            const frameId = ++workerFrameIdRef.current;
            const shouldSyncFrame = syncVideoFrames && sourceMode === "video";
            const modelBitmapPromise = createImageBitmap(modelCanvas);
            const syncedBitmapPromise = shouldSyncFrame ? createImageBitmap(sourceCanvas) : Promise.resolve(undefined);
            void Promise.all([modelBitmapPromise, syncedBitmapPromise]).then(([bitmap, syncedBitmap]) => {
              if (inferenceWorkerRef.current !== activeWorker || workerFrameIdRef.current !== frameId) {
                bitmap.close();
                syncedBitmap?.close();
                workerBusyRef.current = false;
                return;
              }
              pendingSyncedFrameRef.current?.bitmap.close();
              pendingSyncedFrameRef.current = syncedBitmap ? { id: frameId, bitmap: syncedBitmap } : undefined;
              activeWorker.postMessage({
                type: "frame",
                frameId,
                bitmap,
                timestamp,
                detailRecovery,
                skinEdgeCleanup,
                otherSuppression,
                rvmDownsample: rvmDownsample / 100,
                rvmRecurrent,
                rvmForeground,
              }, [bitmap]);
            }).catch((error: unknown) => {
              workerBusyRef.current = false;
              setModelMessage(error instanceof Error ? error.message : "Could not transfer inference frame");
            });
          } else if (imageSegmenter) {
            imageSegmenter.segmentForVideo(modelCanvas, timestamp, (result) => {
              const masks = result.confidenceMasks;
              if (!masks || masks.length === 0) {
                clearInvalidMask();
                setModelMessage("Image Segmenter returned no confidence masks");
                return;
              }

              const maskWidth = masks[0].width;
              const maskHeight = masks[0].height;
              const channelValues = masks.map((mask) => mask.getAsFloat32Array());
              const valueCount = maskWidth * maskHeight;
              if (channelValues.some((values) => values.length !== valueCount)) {
                clearInvalidMask();
                setModelMessage("Image Segmenter returned an invalid mask surface");
                return;
              }

              const values = new Float32Array(valueCount);
              let maximumAcrossChannels = 0;
              for (let index = 0; index < valueCount; index++) {
                if (channelValues.length === 1) {
                  values[index] = channelValues[0][index];
                  maximumAcrossChannels = Math.max(maximumAcrossChannels, values[index]);
                  continue;
                }
                const background = channelValues[0][index];
                const hair = channelValues[1]?.[index] ?? 0;
                const bodySkin = channelValues[2]?.[index] ?? 0;
                const faceSkin = channelValues[3]?.[index] ?? 0;
                let foreground = 0;
                for (let channel = 1; channel < channelValues.length; channel++) {
                  const confidence = channelValues[channel][index];
                  foreground += confidence * (channel === 5 ? 1 - otherSuppression / 100 : 1);
                  maximumAcrossChannels = Math.max(maximumAcrossChannels, confidence);
                }
                maximumAcrossChannels = Math.max(maximumAcrossChannels, background);
                const faceHairRecovery = Math.min(background, (hair + faceSkin) * detailRecovery / 100);
                const ambiguousBodySkin = bodySkin * background * skinEdgeCleanup / 50;
                values[index] = Math.max(0, Math.min(1, foreground + faceHairRecovery - ambiguousBodySkin));
              }
              acceptMask(values, maskWidth, maskHeight, maximumAcrossChannels <= 0.0001);
            });
          } else if (landmarker) {
            landmarker.detectForVideo(modelCanvas, timestamp, (result) => {
              const mask = result.segmentationMasks?.[0];
              if (!mask) {
                clearInvalidMask();
                return;
              }

              const values = new Float32Array(mask.getAsFloat32Array());
              let maximum = 0;
              for (const value of values) maximum = Math.max(maximum, value);
              acceptMask(values, mask.width, mask.height, result.landmarks.length > 0 && maximum <= 0.0001);
            });
          }
          if (!workerInference) statsAccumulatorRef.current.inferenceMs = performance.now() - inferenceStartedAt;
        } catch (error) {
          const detail = error instanceof Error ? error.message : "unknown inference error";
          setModelMessage(`Inference retry · ${detail}`);
        }
      }

      const latestMask = latestMaskRef.current;
      if (latestMask && latestMask.updatedAt !== lastProcessedMaskAt) {
        lastProcessedMaskAt = latestMask.updatedAt;
        const refinementStartedAt = performance.now();
        const previous = previousMaskRef.current?.length === latestMask.values.length
          ? previousMaskRef.current
          : new Float32Array(latestMask.values);
        const stabilizedMask = new Float32Array(latestMask.values.length);
        const isRvm = segmentationModel === "rvm-mobilenetv3";
        const motionCleanupValue = sourceMode === "image" || isRvm ? 0 : motionEdgeCleanup / 100;
        const thresholdValue = Math.max(0.05, threshold / 100 + motionCleanupValue);
        const featherValue = Math.max(0.015, feather / 100 - motionCleanupValue / 2);
        const smoothing = temporalEnabled && !(isRvm && rvmRecurrent) ? temporalSmoothing / 100 : 0;
        let movingPixels = 0;
        for (let index = 0; index < latestMask.values.length; index++) {
          const current = latestMask.values[index];
          const delta = Math.abs(current - previous[index]);
          if (delta >= 0.12) movingPixels++;
          let historyWeight = smoothing;
          if (motionAwareTemporal && historyWeight > 0) {
            historyWeight *= Math.max(0.08, 1 - delta * 2.4);
            historyWeight *= current > previous[index] ? 0.45 : 0.72;
          }
          const stabilized = previous[index] * historyWeight + current * (1 - historyWeight);
          previous[index] = stabilized;
          stabilizedMask[index] = stabilized;
        }
        previousMaskRef.current = previous;

        let refinedMask: Float32Array = stabilizedMask;
        if (!isRvm && spatialSmoothing > 0) {
          refinedMask = edgeAwareRefinement
            ? edgeAwareSmoothMask(
                refinedMask,
                latestMask.width,
                latestMask.height,
                modelContext.getImageData(0, 0, modelWidth, modelHeight),
                spatialSmoothing,
              )
            : boxBlurMask(refinedMask, latestMask.width, latestMask.height, spatialSmoothing);
        }
        if (!isRvm && holeFill) {
          refinedMask = fillEnclosedMaskHoles(refinedMask, latestMask.width, latestMask.height);
        }
        if (edgeExpansion) refinedMask = morphMask(refinedMask, latestMask.width, latestMask.height, edgeExpansion);

        const useGuidedUpsampling = !isRvm && guidedUpsampling > 0;
        const compositeMaskWidth = useGuidedUpsampling ? getGuidedMatteWidth(quality, sourceMode, width) : latestMask.width;
        const compositeMaskHeight = useGuidedUpsampling
          ? Math.round(compositeMaskWidth * height / width)
          : latestMask.height;
        let compositeMask = refinedMask;
        const refinementSource: CanvasImageSource = workerInference && syncVideoFrames && sourceMode === "video" && syncedFrameRef.current
          ? syncedFrameRef.current
          : sourceCanvas;
        if (useGuidedUpsampling) {
          if (guideCanvas.width !== compositeMaskWidth || guideCanvas.height !== compositeMaskHeight) {
            guideCanvas.width = compositeMaskWidth;
            guideCanvas.height = compositeMaskHeight;
          }
          guideContext.imageSmoothingEnabled = true;
          guideContext.imageSmoothingQuality = "high";
          guideContext.drawImage(refinementSource, 0, 0, compositeMaskWidth, compositeMaskHeight);
          compositeMask = guidedUpsampleMask(
            refinedMask,
            latestMask.width,
            latestMask.height,
            guideContext.getImageData(0, 0, compositeMaskWidth, compositeMaskHeight),
            guidedUpsampling,
          );
        } else {
          guideCanvas.width = compositeMaskWidth;
          guideCanvas.height = compositeMaskHeight;
          guideContext.drawImage(refinementSource, 0, 0, compositeMaskWidth, compositeMaskHeight);
        }

        if (maskCanvas.width !== compositeMaskWidth || maskCanvas.height !== compositeMaskHeight) {
          maskCanvas.width = compositeMaskWidth;
          maskCanvas.height = compositeMaskHeight;
          blurMaskCanvas.width = compositeMaskWidth;
          blurMaskCanvas.height = compositeMaskHeight;
        }
        const imageData = maskContext.createImageData(compositeMaskWidth, compositeMaskHeight);
        const blurImageData = effect === "blur" && blurEdgeGuard > 0
          ? blurMaskContext.createImageData(compositeMaskWidth, compositeMaskHeight)
          : undefined;
        const alphaValues = new Float32Array(compositeMask.length);
        let foregroundPixels = 0;
        for (let index = 0; index < compositeMask.length; index++) {
          const alpha = isRvm
            ? Math.max(0, Math.min(1, compositeMask[index]))
            : smoothstep(thresholdValue - featherValue, thresholdValue + featherValue, compositeMask[index]);
          const guardedAlpha = blurImageData
            ? Math.max(alpha, smoothstep(thresholdValue - featherValue - blurEdgeGuard / 100, thresholdValue + featherValue, compositeMask[index]))
            : alpha;
          alphaValues[index] = alpha;
          if (alpha >= 0.5) foregroundPixels++;
          const offset = index * 4;
          imageData.data[offset] = 255;
          imageData.data[offset + 1] = 255;
          imageData.data[offset + 2] = 255;
          imageData.data[offset + 3] = Math.round(alpha * 255);
          if (blurImageData) {
            blurImageData.data[offset] = 255;
            blurImageData.data[offset + 1] = 255;
            blurImageData.data[offset + 2] = 255;
            blurImageData.data[offset + 3] = Math.round(guardedAlpha * 255);
          }
        }
        maskContext.putImageData(imageData, 0, 0);
        if (blurImageData) blurMaskContext.putImageData(blurImageData, 0, 0);
        const replacementGuideCanvas = document.createElement("canvas");
        replacementGuideCanvas.width = compositeMaskWidth;
        replacementGuideCanvas.height = compositeMaskHeight;
        const replacementGuideContext = replacementGuideCanvas.getContext("2d", { alpha: false });
        replacementGuideContext?.drawImage(replacementBackgroundCanvas, 0, 0, compositeMaskWidth, compositeMaskHeight);
        const correction = createEdgeCorrectionCanvas(
          alphaValues,
          compositeMaskWidth,
          compositeMaskHeight,
          guideContext.getImageData(0, 0, compositeMaskWidth, compositeMaskHeight),
          effect === "replace" && replacementGuideContext
            ? replacementGuideContext.getImageData(0, 0, compositeMaskWidth, compositeMaskHeight)
            : undefined,
          edgeDecontamination,
          effect === "replace" ? lightWrap : 0,
        );
        edgeCorrectionCanvas.width = compositeMaskWidth;
        edgeCorrectionCanvas.height = compositeMaskHeight;
        edgeCorrectionContext.clearRect(0, 0, compositeMaskWidth, compositeMaskHeight);
        edgeCorrectionContext.drawImage(correction, 0, 0);
        statsAccumulatorRef.current.refinementMs = performance.now() - refinementStartedAt;
        statsAccumulatorRef.current.coverage = (foregroundPixels / compositeMask.length) * 100;
        statsAccumulatorRef.current.motion = (movingPixels / latestMask.values.length) * 100;
      } else if (!latestMask) {
        statsAccumulatorRef.current.coverage = 0;
        statsAccumulatorRef.current.motion = 0;
      }

      outputContext.save();
      outputContext.clearRect(0, 0, width, height);
      const compositeSource: CanvasImageSource = workerInference && syncVideoFrames && sourceMode === "video" && syncedFrameRef.current
        ? syncedFrameRef.current
        : sourceCanvas;
      const foregroundSource: CanvasImageSource = segmentationModel === "rvm-mobilenetv3" && rvmForeground && rvmForegroundRef.current
        ? rvmForegroundRef.current
        : compositeSource;
      if (effect === "none") {
        outputContext.drawImage(sourceCanvas, 0, 0);
      } else if (effect === "blur") {
        if (blurRenderMode === "efficient") {
          const scale = 0.25;
          const blurWidth = Math.max(1, Math.round(width * scale));
          const blurHeight = Math.max(1, Math.round(height * scale));
          if (backgroundCanvas.width !== blurWidth || backgroundCanvas.height !== blurHeight) {
            backgroundCanvas.width = blurWidth;
            backgroundCanvas.height = blurHeight;
          }
          const scaledRadius = Math.max(2, blurRadius * scale);
          const overscan = Math.ceil(scaledRadius * 1.8);
          backgroundContext.clearRect(0, 0, blurWidth, blurHeight);
          backgroundContext.filter = `blur(${scaledRadius}px) saturate(${backgroundSaturation}%)`;
          backgroundContext.drawImage(compositeSource, -overscan, -overscan, blurWidth + overscan * 2, blurHeight + overscan * 2);
          backgroundContext.filter = "none";
          outputContext.imageSmoothingEnabled = true;
          outputContext.imageSmoothingQuality = "high";
          outputContext.drawImage(backgroundCanvas, 0, 0, width, height);
        } else {
          const overscan = Math.ceil(blurRadius * 1.8);
          outputContext.filter = `blur(${blurRadius}px) saturate(${backgroundSaturation}%)`;
          outputContext.drawImage(compositeSource, -overscan, -overscan, width + overscan * 2, height + overscan * 2);
          outputContext.filter = "none";
        }
      } else {
        outputContext.drawImage(replacementBackgroundCanvas, 0, 0);
      }

      if (effect !== "none" && latestMaskRef.current) {
        foregroundContext.clearRect(0, 0, width, height);
        foregroundContext.globalCompositeOperation = "source-over";
        foregroundContext.drawImage(foregroundSource, 0, 0, width, height);
        foregroundContext.globalCompositeOperation = "destination-in";
        foregroundContext.drawImage(effect === "blur" && blurEdgeGuard > 0 ? blurMaskCanvas : maskCanvas, 0, 0, width, height);
        foregroundContext.globalCompositeOperation = "source-atop";
        foregroundContext.drawImage(edgeCorrectionCanvas, 0, 0, width, height);
        foregroundContext.globalCompositeOperation = "source-over";
        outputContext.drawImage(foregroundCanvas, 0, 0);
      }

      if (previewMode === "mask") {
        outputContext.fillStyle = "#03060a";
        outputContext.fillRect(0, 0, width, height);
        if (latestMaskRef.current) outputContext.drawImage(maskCanvas, 0, 0, width, height);
      } else if (previewMode === "split") {
        const splitX = Math.round((comparePosition / 100) * width);
        outputContext.save();
        outputContext.beginPath();
        outputContext.rect(splitX, 0, width - splitX, height);
        outputContext.clip();
        outputContext.drawImage(sourceCanvas, 0, 0);
        outputContext.restore();
        outputContext.fillStyle = "rgba(255,255,255,.9)";
        outputContext.fillRect(splitX - 1, 0, 2, height);
      }
      outputContext.restore();

      const renderMs = performance.now() - frameStartedAt;
      updateStats(now, renderMs, latestMaskRef.current ? Math.max(0, performance.now() - latestMaskRef.current.updatedAt) : 0);

      animationFrame = requestAnimationFrame(render);
    };

    statsAccumulatorRef.current = { ...statsAccumulatorRef.current, startedAt: performance.now(), frames: 0, masks: 0, dropped: 0 };
    animationFrame = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrame);
  }, [
    adaptiveQuality,
    backgroundSaturation,
    blurEdgeGuard,
    blurRadius,
    blurRenderMode,
    comparePosition,
    clearMaskHistory,
    delegatePreference,
    detailRecovery,
    edgeAwareRefinement,
    edgeDecontamination,
    edgeExpansion,
    effect,
    feather,
    guidedUpsampling,
    inferenceRate,
    imageSegmenter,
    imageViewMode,
    landmarker,
    lightWrap,
    mirror,
    motionEdgeCleanup,
    otherSuppression,
    paused,
    previewMode,
    processingDelegate,
    quality,
    resolution,
    rvmDownsample,
    rvmForeground,
    rvmRecurrent,
    selectedBackground,
    selectedImage.crop,
    segmentationModel,
    settingsHydrated,
    sourceMode,
    sourceReady,
    skinEdgeCleanup,
    syncVideoFrames,
    spatialSmoothing,
    holeFill,
    motionAwareTemporal,
    temporalEnabled,
    temporalSmoothing,
    threshold,
    uploadedSource,
    workerInference,
  ]);

  const resetSettings = () => {
    try {
      window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
    } catch {
      // Reset still restores in-memory defaults if storage is unavailable.
    }
    clearUploadedSource();
    clearCustomBackground();
    stopCamera();
    setPaused(false);
    setVideoPaused(false);
    setCameraError(undefined);
    applySavedSettings(DEFAULT_SETTINGS);
    clearMaskHistory();
  };

  const downloadSnapshot = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `virtual-background-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  return (
    <main className="min-h-screen bg-[#090c11] text-slate-100">
      <img
        ref={imageRef}
        alt=""
        className="hidden"
        crossOrigin="anonymous"
        src={sourceMode === "image" ? activeSource : undefined}
        onLoad={() => setSourceReady((value) => value + 1)}
      />
      <video
        ref={videoRef}
        className="hidden"
        src={sourceMode === "video" ? activeSource : undefined}
        muted
        loop
        playsInline
        autoPlay
        onLoadedData={(event) => {
          setSourceReady((value) => value + 1);
          if (!videoPaused) void event.currentTarget.play().catch(() => setVideoPaused(true));
        }}
      />

      <header className="flex h-14 items-center justify-between border-b border-white/[0.07] bg-[#0d1118]/95 px-4 backdrop-blur-xl lg:px-5">
        <div className="flex items-center gap-3">
          <div className="grid size-8 place-items-center rounded-lg border border-cyan-300/25 bg-cyan-300/10 text-cyan-300">
            <ScanLine className="size-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold tracking-tight text-slate-100">VirtualBackground</h1>
              <Badge className="border-cyan-300/20 bg-cyan-300/10 text-cyan-300">Lab</Badge>
              {settingsHydrated && (
                <Badge
                  className={cn(
                    modifiedSettingCount > 0
                      ? "border-amber-300/25 bg-amber-300/10 text-amber-300"
                      : "border-emerald-300/15 bg-emerald-300/[0.06] text-emerald-400/70",
                  )}
                  aria-live="polite"
                  title={modifiedSettingCount > 0 ? `${modifiedSettingCount} selections differ from defaults and are saved on this device` : "All selections match defaults"}
                >
                  {modifiedSettingCount > 0 ? `Modified · ${modifiedSettingCount} saved` : "Default"}
                </Badge>
              )}
            </div>
            <p className="text-[10px] text-slate-600">On-device segmentation workbench</p>
          </div>
        </div>
        <div className="hidden items-center gap-2 sm:flex">
          <div className="flex items-center gap-2 rounded-lg border border-white/[0.07] bg-black/20 px-3 py-1.5 text-[11px] text-slate-400">
            <span className={cn("size-1.5 rounded-full", modelStatus === "ready" ? "bg-emerald-400 shadow-[0_0_10px_#34d399]" : modelStatus === "error" ? "bg-rose-400" : "animate-pulse bg-amber-400")} />
            {modelMessage}
          </div>
          <Button
            variant="secondary"
            size="sm"
            className={cn(modifiedSettingCount > 0 && "border-amber-300/25 text-amber-200")}
            disabled={modifiedSettingCount === 0 && !uploadedSource && !customBackgroundUrl}
            onClick={resetSettings}
            title="Restore defaults and clear saved selections"
          >
            <RotateCcw className="size-3.5" /> Reset
          </Button>
          <Button size="sm" onClick={downloadSnapshot}>
            <Download className="size-3.5" /> Snapshot
          </Button>
        </div>
      </header>

      <div className="workspace-grid">
        <aside className="source-panel border-r border-white/[0.07] bg-[#0d1118] p-3">
          <div className="mb-3 flex items-center gap-2 px-1 text-[10px] font-semibold uppercase tracking-[.16em] text-slate-600">
            <MonitorUp className="size-3.5" /> Input source
          </div>
          <Tabs value={sourceMode} onValueChange={selectSourceMode}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="image"><ImageIcon className="size-3.5" />Image</TabsTrigger>
              <TabsTrigger value="video"><Video className="size-3.5" />Video</TabsTrigger>
              <TabsTrigger value="camera"><Camera className="size-3.5" />Cam</TabsTrigger>
            </TabsList>
            <TabsContent value="image" className="space-y-2">
              <div className="grid grid-cols-2 gap-1 rounded-lg border border-white/[0.07] bg-black/20 p-1" aria-label="Image benchmark layout">
                {(["single", "grid"] as ImageViewMode[]).map((value) => (
                  <button
                    key={value}
                    onClick={() => selectImageViewMode(value)}
                    className={cn("rounded-md px-2 py-2 text-[10px] font-medium text-slate-500 transition", imageViewMode === value && "bg-cyan-300/12 text-cyan-300 ring-1 ring-cyan-300/20")}
                  >
                    {value === "single" ? "Single image" : "4-up benchmark"}
                  </button>
                ))}
              </div>
              {IMAGE_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => { clearUploadedSource(); clearMaskHistory(); setImageViewMode("single"); setImagePreset(preset.id); }}
                  className={cn("source-tile", imageViewMode === "single" && !uploadedSource && imagePreset === preset.id && "source-tile-active")}
                >
                  <span className="h-12 w-[74px] shrink-0 rounded-md bg-cover bg-center" style={{ backgroundImage: `url(${preset.src})` }} />
                  <span className="min-w-0 text-left">
                    <span className="block truncate text-xs font-medium text-slate-200">{preset.name}</span>
                    <span className="block truncate text-[10px] text-slate-600">{preset.detail}</span>
                  </span>
                </button>
              ))}
            </TabsContent>
            <TabsContent value="video" className="space-y-2">
              {VIDEO_PRESETS.map((preset, index) => (
                <button
                  key={preset.id}
                  onClick={() => { clearUploadedSource(); clearMaskHistory(); setVideoPreset(preset.id); }}
                  className={cn("source-tile", !uploadedSource && videoPreset === preset.id && "source-tile-active")}
                >
                  <span className={cn("grid h-12 w-[74px] shrink-0 place-items-center rounded-md", index ? "bg-gradient-to-br from-violet-900 to-slate-950" : "bg-gradient-to-br from-cyan-900 to-slate-950")}>
                    <CirclePlay className="size-5 text-white/75" />
                  </span>
                  <span className="min-w-0 text-left">
                    <span className="block truncate text-xs font-medium text-slate-200">{preset.name}</span>
                    <span className="block truncate text-[10px] text-slate-600">{preset.detail}</span>
                  </span>
                </button>
              ))}
              <Button variant="secondary" size="sm" className="w-full" onClick={() => void toggleVideoPlayback()}>
                {videoPaused ? <><CirclePlay className="size-3.5" /> Play motion</> : <><CirclePause className="size-3.5" /> Pause motion</>}
              </Button>
            </TabsContent>
            <TabsContent value="camera">
              <div className="rounded-xl border border-dashed border-white/10 bg-black/20 p-4 text-center">
                <div className="mx-auto mb-3 grid size-10 place-items-center rounded-full bg-cyan-300/10 text-cyan-300"><Camera className="size-4" /></div>
                <p className="text-xs font-medium text-slate-300">Live camera input</p>
                <p className="mt-1 text-[10px] leading-relaxed text-slate-600">Frames stay on this device and are processed locally.</p>
                <Button className="mt-4 w-full" size="sm" onClick={cameraActive ? stopCamera : startCamera}>
                  {cameraActive ? <><CirclePause className="size-3.5" /> Stop camera</> : <><Camera className="size-3.5" /> Enable camera</>}
                </Button>
                {cameraError && <p className="mt-2 text-[10px] text-rose-400">{cameraError}</p>}
              </div>
            </TabsContent>
          </Tabs>

          <label className="mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-white/10 px-3 py-2.5 text-xs text-slate-500 transition hover:border-cyan-300/30 hover:text-cyan-300">
            <Upload className="size-3.5" /> Upload image or video
            <input className="hidden" type="file" accept="image/*,video/*" onChange={handleSourceUpload} />
          </label>
          {uploadedSource && <p className="mt-2 truncate px-1 text-[10px] text-cyan-400">{uploadedSource.name}</p>}

          <div className="my-4 h-px bg-white/[0.07]" />
          <div className="mb-3 flex items-center gap-2 px-1 text-[10px] font-semibold uppercase tracking-[.16em] text-slate-600">
            <Layers3 className="size-3.5" /> Backgrounds
          </div>
          <div className="grid grid-cols-2 gap-2">
            {BACKGROUNDS.map((preset) => (
              <button
                key={preset.id}
                onClick={() => { setBackground(preset.id); clearCustomBackground(); setEffect("replace"); }}
                className={cn("background-tile", effect === "replace" && !customBackgroundUrl && background === preset.id && "background-tile-active")}
              >
                <span
                  className="block aspect-video rounded-md bg-cover bg-center"
                  style={"src" in preset ? { backgroundImage: `url(${preset.src})` } : { background: `linear-gradient(135deg, ${preset.colors.join(",")})` }}
                />
                <span className="mt-1.5 block truncate text-[10px] text-slate-500">{preset.name}</span>
              </button>
            ))}
          </div>
          <label className="mt-2 flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-[11px] text-slate-400 hover:bg-white/[0.06]">
            <ImageIcon className="size-3.5" /> Custom background
            <input className="hidden" type="file" accept="image/*" onChange={handleBackgroundUpload} />
          </label>
        </aside>

        <section className="min-w-0 bg-[#090c11] p-3 lg:p-4">
          <div className="mx-auto flex h-full max-w-[1320px] flex-col gap-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-sm font-semibold">Processed output</h2>
                  <Badge className="border-emerald-400/20 bg-emerald-400/10 text-emerald-300">Live</Badge>
                </div>
                <p className="mt-0.5 text-[10px] text-slate-600">{RESOLUTIONS[resolution].width} × {RESOLUTIONS[resolution].height} · {sourceMode === "image" && imageViewMode === "grid" ? "4 independent mattes" : `${inferenceRate} mask fps`} · {quality}</p>
              </div>
              <div className="flex items-center gap-1 rounded-lg border border-white/[0.07] bg-[#11161f] p-1">
                {(["result", "mask", "split"] as PreviewMode[]).map((mode) => (
                  <button key={mode} onClick={() => setPreviewMode(mode)} className={cn("rounded-md px-2.5 py-1.5 text-[10px] font-medium capitalize text-slate-500", previewMode === mode && "bg-white/10 text-slate-100")}>{mode}</button>
                ))}
              </div>
            </div>

            <Card className="preview-card relative min-h-0 flex-1 overflow-hidden bg-black">
              <canvas ref={canvasRef} className="h-full w-full object-contain" />
              <div className="pointer-events-none absolute left-3 top-3 flex gap-2">
                <Badge>{resolution}</Badge>
                {sourceMode === "image" && imageViewMode === "grid" && <Badge className="border-violet-300/20 text-violet-300">4-up</Badge>}
                <Badge className="border-cyan-300/20 text-cyan-300">{selectedSegmentationModel.name}</Badge>
              </div>
              <div className="absolute right-3 top-3 flex gap-2">
                <Button variant="secondary" size="icon" className="size-8 bg-black/40" onClick={() => setPaused((value) => !value)} aria-label={paused ? "Resume processing" : "Pause processing"}>
                  {paused ? <CirclePlay className="size-3.5" /> : <CirclePause className="size-3.5" />}
                </Button>
                <Button variant="secondary" size="icon" className="size-8 bg-black/40" onClick={() => canvasRef.current?.requestFullscreen()} aria-label="Open fullscreen">
                  <Maximize2 className="size-3.5" />
                </Button>
              </div>
              {previewMode === "split" && (
                <div className="absolute inset-x-6 bottom-4 rounded-lg border border-white/10 bg-black/55 px-3 py-2 backdrop-blur">
                  <div className="mb-1.5 flex justify-between text-[9px] uppercase tracking-[.12em] text-slate-400"><span>Processed</span><span>Original</span></div>
                  <Slider value={[comparePosition]} min={10} max={90} step={1} onValueChange={([value]) => setComparePosition(value)} />
                </div>
              )}
              {modelStatus === "loading" && (
                <div className="absolute inset-0 grid place-items-center bg-black/55 backdrop-blur-sm">
                  <div className="text-center"><Aperture className="mx-auto size-7 animate-spin text-cyan-300" /><p className="mt-3 text-xs text-slate-300">Preparing on-device model…</p></div>
                </div>
              )}
            </Card>

            <div className="grid grid-cols-2 gap-2 md:grid-cols-5 xl:grid-cols-9">
              <Metric label="Output" value={stats.fps.toFixed(1)} unit="fps" />
              <Metric label="Matte rate" value={sourceMode === "image" ? imageViewMode === "grid" ? "4 fixed" : "fixed" : stats.maskFps.toFixed(1)} unit={sourceMode === "image" ? "" : "fps"} />
              <Metric label="Inference" value={stats.inferenceMs.toFixed(1)} unit="ms" tone="violet" />
              <Metric label="Refinement" value={stats.refinementMs.toFixed(1)} unit="ms" tone="violet" />
              <Metric label="Composite" value={stats.renderMs.toFixed(1)} unit="ms" tone="lime" />
              <Metric label="Coverage" value={stats.maskCoverage.toFixed(1)} unit="%" />
              <Metric label="Motion" value={stats.maskMotion.toFixed(1)} unit="%" tone="lime" />
              <Metric label="Matte age" value={sourceMode === "image" ? "fixed" : stats.matteAgeMs.toFixed(0)} unit={sourceMode === "image" ? "" : "ms"} />
              <Metric label="Heap" value={stats.memoryMb ? stats.memoryMb.toFixed(0) : "—"} unit={stats.memoryMb ? "MB" : ""} tone="violet" />
            </div>

            <Card className="hidden h-20 items-center gap-4 px-4 md:flex">
              <div className="min-w-24">
                <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[.14em] text-slate-500"><Activity className="size-3.5 text-cyan-400" /> Frame health</div>
                <div className="mt-1 font-mono text-xs text-slate-300">{stats.dropped} slow frames</div>
              </div>
              <div className="flex h-10 flex-1 items-end gap-1">
                {frameHistory.map((value, index) => <span key={index} className="min-w-1 flex-1 rounded-sm bg-cyan-400/70 transition-all" style={{ height: `${Math.max(8, value * 100)}%`, opacity: 0.3 + value * 0.7 }} />)}
              </div>
              <div className="grid grid-cols-3 gap-5 border-l border-white/[0.07] pl-5 text-center">
                <div><div className="font-mono text-xs text-slate-300">{processingDelegate}</div><div className="text-[9px] text-slate-600">delegate</div></div>
                <div><div className="font-mono text-xs text-slate-300">{selectedSegmentationModel.name}</div><div className="text-[9px] text-slate-600">model</div></div>
                <div><div className="font-mono text-xs text-emerald-300">LOCAL</div><div className="text-[9px] text-slate-600">privacy</div></div>
              </div>
            </Card>
          </div>
        </section>

        <aside className="settings-panel border-l border-white/[0.07] bg-[#0d1118] p-3">
          <Tabs value={settingsTab} onValueChange={(value) => setSettingsTab(value as SettingsTab)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="effect"><WandSparkles className="size-3.5" />Effect</TabsTrigger>
              <TabsTrigger value="mask"><ScanLine className="size-3.5" />Mask</TabsTrigger>
              <TabsTrigger value="perf"><Gauge className="size-3.5" />Perf</TabsTrigger>
            </TabsList>

            <TabsContent value="effect">
              <div className="mb-1 grid grid-cols-3 gap-1 rounded-lg border border-white/[0.07] bg-black/20 p-1">
                {(["none", "blur", "replace"] as EffectMode[]).map((mode) => (
                  <button key={mode} onClick={() => setEffect(mode)} className={cn("rounded-md px-2 py-2 text-[10px] font-medium capitalize text-slate-500 transition", effect === mode && "bg-cyan-300/12 text-cyan-300 ring-1 ring-cyan-300/20")}>{mode}</button>
                ))}
              </div>
              <SettingRow label="Blur radius" value={`${blurRadius}px`}><Slider value={[blurRadius]} min={2} max={64} step={1} onValueChange={([value]) => setBlurRadius(value)} /></SettingRow>
              <SettingRow label="Background saturation" value={`${backgroundSaturation}%`}><Slider value={[backgroundSaturation]} min={40} max={130} step={1} onValueChange={([value]) => setBackgroundSaturation(value)} /></SettingRow>
              <SettingRow label="Blur edge guard" value={`${blurEdgeGuard}%`}><Slider value={[blurEdgeGuard]} min={0} max={24} step={1} onValueChange={([value]) => setBlurEdgeGuard(value)} /></SettingRow>
              <SettingRow label="Edge color cleanup" value={`${edgeDecontamination}%`}><Slider value={[edgeDecontamination]} min={0} max={100} step={5} onValueChange={([value]) => setEdgeDecontamination(value)} /></SettingRow>
              <SettingRow label="Replacement light wrap" value={background === "black" && !customBackgroundUrl ? "off on black" : `${lightWrap}%`}><Slider value={[lightWrap]} min={0} max={30} step={1} onValueChange={([value]) => setLightWrap(value)} /></SettingRow>
              <div className="space-y-2 py-3">
                <div className="text-xs font-medium text-slate-300">Blur renderer</div>
                <div className="grid grid-cols-2 gap-1">
                  {(["efficient", "full"] as BlurRenderMode[]).map((value) => (
                    <button key={value} onClick={() => setBlurRenderMode(value)} className={cn("rounded-md border border-white/[0.07] px-2 py-2 text-[10px] capitalize text-slate-500", blurRenderMode === value && "border-cyan-300/20 bg-cyan-300/10 text-cyan-300")}>{value === "efficient" ? "Scaled HQ" : "Full resolution"}</button>
                  ))}
                </div>
              </div>
              <div className="setting-toggle"><div><p>Mirror camera</p><span>Match meeting preview</span></div><Switch checked={mirror} onCheckedChange={setMirror} /></div>
              <div className="setting-toggle"><div><p>Adaptive quality</p><span>Protect output frame rate</span></div><Switch checked={adaptiveQuality} onCheckedChange={setAdaptiveQuality} /></div>
            </TabsContent>

            <TabsContent value="mask">
              <SettingRow label="Foreground threshold" value={`${threshold}%`}><Slider value={[threshold]} min={10} max={85} step={1} onValueChange={([value]) => setThreshold(value)} /></SettingRow>
              <SettingRow label="Edge feather" value={`${feather}%`}><Slider value={[feather]} min={1} max={40} step={1} onValueChange={([value]) => setFeather(value)} /></SettingRow>
              <SettingRow label="Temporal smoothing" value={`${temporalSmoothing}%`}><Slider value={[temporalSmoothing]} min={0} max={85} step={1} onValueChange={([value]) => setTemporalSmoothing(value)} /></SettingRow>
              <SettingRow label="Spatial smoothing" value={`${spatialSmoothing}px`}><Slider value={[spatialSmoothing]} min={0} max={3} step={1} onValueChange={([value]) => setSpatialSmoothing(value)} /></SettingRow>
              <SettingRow label="Guided edge upsample" value={segmentationModel === "rvm-mobilenetv3" ? "native" : `${guidedUpsampling}%`}><Slider value={[guidedUpsampling]} min={0} max={100} step={5} disabled={segmentationModel === "rvm-mobilenetv3"} onValueChange={([value]) => setGuidedUpsampling(value)} /></SettingRow>
              <SettingRow label="Matte expansion" value={`${edgeExpansion}px`}><Slider value={[edgeExpansion]} min={-3} max={4} step={1} onValueChange={([value]) => setEdgeExpansion(value)} /></SettingRow>
              <SettingRow label="Hair + face recovery" value={`${detailRecovery}%`}><Slider value={[detailRecovery]} min={0} max={35} step={1} onValueChange={([value]) => { clearMaskHistory(); setDetailRecovery(value); }} /></SettingRow>
              <SettingRow label="Skin boundary cleanup" value={`${skinEdgeCleanup}%`}><Slider value={[skinEdgeCleanup]} min={0} max={50} step={2} onValueChange={([value]) => { clearMaskHistory(); setSkinEdgeCleanup(value); }} /></SettingRow>
              <SettingRow label="Other-class suppression" value={`${otherSuppression}%`}><Slider value={[otherSuppression]} min={0} max={80} step={5} onValueChange={([value]) => { clearMaskHistory(); setOtherSuppression(value); }} /></SettingRow>
              <SettingRow label="Motion edge cleanup" value={`${motionEdgeCleanup}%`}><Slider value={[motionEdgeCleanup]} min={0} max={10} step={1} onValueChange={([value]) => setMotionEdgeCleanup(value)} /></SettingRow>
              <div className="setting-toggle"><div><p>Temporal stage</p><span>Stabilize confidence over time</span></div><Switch checked={temporalEnabled} onCheckedChange={setTemporalEnabled} /></div>
              <div className="setting-toggle"><div><p>Motion-aware response</p><span>Release hands and limbs faster</span></div><Switch checked={motionAwareTemporal} onCheckedChange={setMotionAwareTemporal} /></div>
              <div className="setting-toggle"><div><p>Edge-aware refinement</p><span>Smooth the matte without crossing image edges</span></div><Switch checked={edgeAwareRefinement} onCheckedChange={setEdgeAwareRefinement} /></div>
              <div className="setting-toggle"><div><p>Fill mask holes</p><span>Fill enclosed defects without closing finger gaps</span></div><Switch checked={holeFill} onCheckedChange={setHoleFill} /></div>
              <div className="rounded-lg border border-cyan-300/10 bg-cyan-300/[0.04] p-3 text-[10px] leading-relaxed text-slate-500">
                <div className="mb-1 flex items-center gap-2 font-medium text-cyan-300"><Sparkles className="size-3.5" /> Tuning hint</div>
                {segmentationModel === "rvm-mobilenetv3"
                  ? "RVM supplies a native soft alpha and its own recurrent temporal state, so threshold, spatial smoothing, and hole filling are bypassed. Matte expansion and compositing controls remain active."
                  : "Selfie Multiclass is the quality default. Face and hair recovery is separated from stricter body-skin boundaries; chroma-guided upsampling and edge-color cleanup target finger spill without eroding the face."}
              </div>
            </TabsContent>

            <TabsContent value="perf">
              <SettingRow label="Mask inference rate" value={`${inferenceRate} fps`}><Slider value={[inferenceRate]} min={4} max={30} step={1} onValueChange={([value]) => setInferenceRate(value)} /></SettingRow>
              <div className="setting-toggle"><div><p>Worker inference</p><span>{imageViewMode === "grid" && sourceMode === "image" ? "Required by the four-up benchmark" : segmentationModel === "rvm-mobilenetv3" ? "Required by the experimental RVM path" : "Keep model execution off the render thread"}</span></div><Switch checked={workerInference} disabled={segmentationModel === "rvm-mobilenetv3" || (imageViewMode === "grid" && sourceMode === "image")} onCheckedChange={(value) => { clearMaskHistory(); setWorkerInference(value); }} /></div>
              <div className="setting-toggle"><div><p>Frame-synced video</p><span>Composite each matte with its captured frame</span></div><Switch checked={syncVideoFrames} onCheckedChange={(value) => { clearMaskHistory(); setSyncVideoFrames(value); }} /></div>
              <div className="space-y-2 py-3">
                <div className="text-xs font-medium text-slate-300">Segmentation model</div>
                <div className="grid grid-cols-2 gap-1">
                  {SEGMENTATION_MODEL_ORDER.map((value) => (
                    <button key={value} onClick={() => selectSegmentationModel(value)} className={cn("rounded-md border border-white/[0.07] px-2 py-2 text-[10px] text-slate-500", segmentationModel === value && "border-cyan-300/20 bg-cyan-300/10 text-cyan-300")}><span className="block">{SEGMENTATION_MODELS[value].name}</span><span className="mt-0.5 block text-[8px] opacity-60">{SEGMENTATION_MODELS[value].detail}</span></button>
                  ))}
                </div>
              </div>
              {segmentationModel === "rvm-mobilenetv3" && (
                <div className="space-y-1 rounded-lg border border-violet-300/15 bg-violet-300/[0.04] p-3">
                  <div className="mb-2 flex items-center justify-between"><span className="text-xs font-medium text-violet-300">RVM experiment</span><Badge className="border-violet-300/20 text-violet-300">GPL-3.0</Badge></div>
                  <SettingRow label="Internal downsample" value={`${rvmDownsample}%`}><Slider value={[rvmDownsample]} min={25} max={100} step={5} onValueChange={([value]) => { clearMaskHistory(); setRvmDownsample(value); }} /></SettingRow>
                  <div className="setting-toggle"><div><p>Recurrent state</p><span>Carry temporal memory between video frames</span></div><Switch checked={rvmRecurrent} onCheckedChange={(value) => { clearMaskHistory(); setRvmRecurrent(value); }} /></div>
                  <div className="setting-toggle"><div><p>Predicted foreground</p><span>Use RVM color reconstruction for the person layer</span></div><Switch checked={rvmForeground} onCheckedChange={(value) => { clearMaskHistory(); setRvmForeground(value); }} /></div>
                  <p className="pt-2 text-[9px] leading-relaxed text-slate-600">Experimental MobileNetV3 TensorFlow.js model. Recurrent state resets when the source changes or a video loops. Review GPL-3.0 compatibility before distribution.</p>
                </div>
              )}
              <div className="space-y-2 py-3">
                <div className="text-xs font-medium text-slate-300">Delegate</div>
                <div className="grid grid-cols-3 gap-1">
                  {(["auto", "GPU", "CPU"] as DelegatePreference[]).map((value) => (
                    <button key={value} onClick={() => selectDelegatePreference(value)} className={cn("rounded-md border border-white/[0.07] px-2 py-2 text-[10px] text-slate-500", delegatePreference === value && "border-cyan-300/20 bg-cyan-300/10 text-cyan-300")}>{value === "auto" ? "Auto" : value}</button>
                  ))}
                </div>
              </div>
              <div className="space-y-2 py-3">
                <div className="text-xs font-medium text-slate-300">Model input</div>
                <div className="grid grid-cols-3 gap-1">
                  {(["fast", "balanced", "quality"] as Quality[]).map((value) => (
                    <button key={value} onClick={() => { clearMaskHistory(); setQuality(value); }} className={cn("rounded-md border border-white/[0.07] px-2 py-2 text-[10px] capitalize text-slate-500", quality === value && "border-cyan-300/20 bg-cyan-300/10 text-cyan-300")}>{value}</button>
                  ))}
                </div>
              </div>
              <div className="space-y-2 py-3">
                <div className="text-xs font-medium text-slate-300">Output resolution</div>
                <div className="grid grid-cols-3 gap-1">
                  {(Object.keys(RESOLUTIONS) as (keyof typeof RESOLUTIONS)[]).map((value) => (
                    <button key={value} onClick={() => setResolution(value)} className={cn("rounded-md border border-white/[0.07] px-2 py-2 text-[10px] text-slate-500", resolution === value && "border-cyan-300/20 bg-cyan-300/10 text-cyan-300")}>{value}</button>
                  ))}
                </div>
              </div>
              <div className="mt-2 space-y-2 rounded-lg border border-white/[0.07] bg-black/20 p-3">
                <div className="flex items-center justify-between text-[10px]"><span className="flex items-center gap-2 text-slate-500"><Cpu className="size-3.5" /> Model</span><span className="font-mono text-slate-300">{selectedSegmentationModel.name}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Inference input</span><span className="font-mono text-slate-300">{getModelInputWidth(segmentationModel, quality)}×{Math.round((getModelInputWidth(segmentationModel, quality) * 9) / 16)}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Mask surface</span><span className="font-mono text-slate-300">{maskSurface}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Composite matte</span><span className="font-mono text-slate-300">{segmentationModel !== "rvm-mobilenetv3" && guidedUpsampling > 0 ? `${getGuidedMatteWidth(quality, sourceMode, RESOLUTIONS[resolution].width)}×${Math.round(getGuidedMatteWidth(quality, sourceMode, RESOLUTIONS[resolution].width) * RESOLUTIONS[resolution].height / RESOLUTIONS[resolution].width)}` : maskSurface}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Active delegate</span><span className="font-mono text-slate-300">{processingDelegate}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Execution</span><span className="font-mono text-slate-300">{workerInference ? "Worker" : "Main thread"}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Matte processing</span><span className="font-mono text-slate-300">{segmentationModel === "rvm-mobilenetv3" ? "Native soft alpha" : "Refined confidence"}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Video composite</span><span className="font-mono text-slate-300">{workerInference && syncVideoFrames ? "Frame synced" : "Latest frame"}</span></div>
                <div className="flex items-center justify-between text-[10px]"><span className="text-slate-500">Dropped</span><span className="font-mono text-slate-300">{stats.dropped} / window</span></div>
              </div>
            </TabsContent>
          </Tabs>

          <Card className="mt-3 bg-gradient-to-br from-cyan-300/[0.06] to-transparent">
            <CardHeader className="pb-2"><div className="flex items-center gap-2 text-xs font-medium"><Settings2 className="size-3.5 text-cyan-300" /> Pipeline</div><Badge>Web</Badge></CardHeader>
            <CardContent>
              <div className="pipeline-list">
                <span>Source capture</span><i />
                <span>{selectedSegmentationModel.name} · {workerInference ? "worker" : "main"}</span><i />
                <span>{workerInference && syncVideoFrames && sourceMode === "video" ? "Frame/matte sync" : "Latest-frame composite"}</span><i />
                <span>{segmentationModel === "rvm-mobilenetv3" ? rvmRecurrent ? "RVM recurrent temporal state" : "RVM stateless inference" : temporalEnabled ? "Adaptive temporal matte" : "Temporal bypass"}</span><i />
                <span>{segmentationModel === "rvm-mobilenetv3" ? "Native soft alpha" : guidedUpsampling ? "Guided high-res matte" : spatialSmoothing || holeFill ? edgeAwareRefinement ? "Edge-aware refinement" : "Spatial refinement" : "Spatial bypass"}</span><i />
                <span>{effect !== "none" && edgeDecontamination ? effect === "replace" && lightWrap && (background !== "black" || customBackgroundUrl) ? "Edge cleanup + RGB light wrap" : "Edge color cleanup" : effect === "blur" && blurEdgeGuard ? "Guarded blur composite" : "Canvas composite"}</span>
              </div>
            </CardContent>
          </Card>
        </aside>
      </div>
    </main>
  );
}
