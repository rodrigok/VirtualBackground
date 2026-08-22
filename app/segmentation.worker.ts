import type {
  ImageSegmenter as ImageSegmenterInstance,
  PoseLandmarker as PoseLandmarkerInstance,
} from "@mediapipe/tasks-vision";
import type { GraphModel, Tensor } from "@tensorflow/tfjs";

type SegmentationTask = "selfie" | "pose" | "rvm";

type InitMessage = {
  type: "init";
  modelPath: string;
  task: SegmentationTask;
  delegate: "GPU" | "CPU";
};

type FrameMessage = {
  type: "frame";
  frameId: number;
  benchmarkId?: string;
  bitmap: ImageBitmap;
  timestamp: number;
  detailRecovery: number;
  skinEdgeCleanup: number;
  otherSuppression: number;
  rvmDownsample: number;
  rvmRecurrent: boolean;
  rvmForeground: boolean;
};

type ResetMessage = { type: "reset" };
type WorkerMessage = InitMessage | FrameMessage | ResetMessage;

let imageSegmenter: ImageSegmenterInstance | undefined;
let poseLandmarker: PoseLandmarkerInstance | undefined;
let rvmModel: GraphModel | undefined;
let tfModule: typeof import("@tensorflow/tfjs") | undefined;
let recurrentStates: [Tensor, Tensor, Tensor, Tensor] | undefined;
let rvmProcessing = false;
let rvmResetPending = false;
let activeTask: SegmentationTask = "selfie";

const respond = (message: object, transfer?: Transferable[]) => {
  globalThis.postMessage(message, { transfer });
};

const disposeRecurrentStates = () => {
  recurrentStates?.forEach((tensor) => tensor.dispose());
  recurrentStates = undefined;
};

const resetRecurrentStates = () => {
  disposeRecurrentStates();
  if (!tfModule) return;
  recurrentStates = [tfModule.scalar(0), tfModule.scalar(0), tfModule.scalar(0), tfModule.scalar(0)];
};

const requestRecurrentReset = () => {
  if (rvmProcessing) {
    rvmResetPending = true;
    return;
  }
  resetRecurrentStates();
};

const closeModel = () => {
  imageSegmenter?.close();
  poseLandmarker?.close();
  rvmModel?.dispose();
  disposeRecurrentStates();
  imageSegmenter = undefined;
  poseLandmarker = undefined;
  rvmModel = undefined;
  rvmResetPending = false;
};

const initialize = async (message: InitMessage) => {
  closeModel();
  activeTask = message.task;
  if (message.task === "rvm") {
    tfModule = await import("@tensorflow/tfjs");
    const requestedBackend = message.delegate === "GPU" ? "webgl" : "cpu";
    const backendAvailable = await tfModule.setBackend(requestedBackend);
    if (!backendAvailable) throw new Error(`TensorFlow.js ${requestedBackend} backend is unavailable`);
    await tfModule.ready();
    rvmModel = await tfModule.loadGraphModel(message.modelPath);
    resetRecurrentStates();
    respond({ type: "ready", backend: tfModule.getBackend() });
    return;
  }

  const visionModuleUrl = "/mediapipe/vision_bundle.mjs";
  const importUnbundled = new Function("url", "return import(url)") as (
    url: string,
  ) => Promise<typeof import("@mediapipe/tasks-vision")>;
  const { FilesetResolver, ImageSegmenter, PoseLandmarker } = await importUnbundled(visionModuleUrl);
  const files = await FilesetResolver.forVisionTasks("/mediapipe/wasm", true);
  if (message.task === "selfie") {
    imageSegmenter = await ImageSegmenter.createFromOptions(files, {
      baseOptions: { modelAssetPath: message.modelPath, delegate: message.delegate },
      runningMode: "VIDEO",
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
  } else {
    poseLandmarker = await PoseLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: message.modelPath, delegate: message.delegate },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.55,
      outputSegmentationMasks: true,
    });
  }
  respond({ type: "ready", backend: message.delegate.toLowerCase() });
};

const createForegroundBitmap = async (foreground: Tensor) => {
  if (!tfModule) return undefined;
  const rgba = tfModule.tidy(() => {
    const rgb = foreground.squeeze([0]).clipByValue(0, 1).mul(255).cast("int32");
    const [height, width] = rgb.shape;
    return tfModule!.concat([rgb, tfModule!.fill([height, width, 1], 255, "int32")], -1);
  });
  const [height, width] = rgba.shape;
  const pixels = new Uint8ClampedArray(await rgba.data());
  rgba.dispose();
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d");
  if (!context) return undefined;
  context.putImageData(new ImageData(pixels, width, height), 0, 0);
  return canvas.transferToImageBitmap();
};

const processRvmFrame = async (message: FrameMessage) => {
  if (!tfModule || !rvmModel || !recurrentStates) throw new Error("RVM is not initialized");
  const startedAt = performance.now();
  rvmProcessing = true;
  const source = tfModule.tidy(() => tfModule!.browser.fromPixels(message.bitmap).toFloat().div(255).expandDims(0));
  const downsampleRatio = tfModule.scalar(message.rvmDownsample);
  const inputStates = recurrentStates;
  try {
    const output = await rvmModel.executeAsync(
      {
        src: source,
        r1i: inputStates[0],
        r2i: inputStates[1],
        r3i: inputStates[2],
        r4i: inputStates[3],
        downsample_ratio: downsampleRatio,
      },
      ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
    );
    const [foreground, alpha, ...nextStates] = output as Tensor[];
    const values = new Float32Array(await alpha.data());
    const foregroundBitmap = message.rvmForeground ? await createForegroundBitmap(foreground) : undefined;
    const [, height, width] = alpha.shape;

    foreground.dispose();
    alpha.dispose();
    inputStates.forEach((tensor) => tensor.dispose());
    if (message.rvmRecurrent && !rvmResetPending) {
      recurrentStates = nextStates as [Tensor, Tensor, Tensor, Tensor];
    } else {
      nextStates.forEach((tensor) => tensor.dispose());
      recurrentStates = undefined;
      resetRecurrentStates();
    }
    rvmResetPending = false;

    const response = {
      type: "mask",
      frameId: message.frameId,
      benchmarkId: message.benchmarkId,
      values: values.buffer,
      width,
      height,
      inferenceMs: performance.now() - startedAt,
      unreadable: false,
      foreground: foregroundBitmap,
    };
    const transfer: Transferable[] = [values.buffer];
    if (foregroundBitmap) transfer.push(foregroundBitmap);
    respond(response, transfer);
  } finally {
    rvmProcessing = false;
    source.dispose();
    downsampleRatio.dispose();
    message.bitmap.close();
    if (rvmResetPending) {
      rvmResetPending = false;
      resetRecurrentStates();
    }
  }
};

const processMediaPipeFrame = (message: FrameMessage) => {
  const startedAt = performance.now();
  try {
    if (activeTask === "selfie" && imageSegmenter) {
      imageSegmenter.segmentForVideo(message.bitmap, message.timestamp, (result) => {
        const masks = result.confidenceMasks;
        if (!masks?.length) throw new Error("Image Segmenter returned no confidence masks");
        const width = masks[0].width;
        const height = masks[0].height;
        const channels = masks.map((mask) => mask.getAsFloat32Array());
        const valueCount = width * height;
        if (channels.some((values) => values.length !== valueCount)) throw new Error("Image Segmenter returned an invalid mask surface");

        const values = new Float32Array(valueCount);
        let maximum = 0;
        for (let index = 0; index < valueCount; index++) {
          if (channels.length === 1) {
            values[index] = channels[0][index];
            maximum = Math.max(maximum, values[index]);
            continue;
          }
          const background = channels[0][index];
          const hair = channels[1]?.[index] ?? 0;
          const bodySkin = channels[2]?.[index] ?? 0;
          const faceSkin = channels[3]?.[index] ?? 0;
          let foreground = 0;
          for (let channel = 1; channel < channels.length; channel++) {
            const confidence = channels[channel][index];
            foreground += confidence * (channel === 5 ? 1 - message.otherSuppression / 100 : 1);
            maximum = Math.max(maximum, confidence);
          }
          const faceHairRecovery = Math.min(background, (hair + faceSkin) * message.detailRecovery / 100);
          const ambiguousBodySkin = bodySkin * background * message.skinEdgeCleanup / 50;
          values[index] = Math.max(0, Math.min(1, foreground + faceHairRecovery - ambiguousBodySkin));
        }
        respond(
          { type: "mask", frameId: message.frameId, benchmarkId: message.benchmarkId, values: values.buffer, width, height, inferenceMs: performance.now() - startedAt, unreadable: maximum <= 0.0001 },
          [values.buffer],
        );
      });
    } else if (poseLandmarker) {
      poseLandmarker.detectForVideo(message.bitmap, message.timestamp, (result) => {
        const mask = result.segmentationMasks?.[0];
        if (!mask) throw new Error("Pose Landmarker returned no segmentation mask");
        const values = new Float32Array(mask.getAsFloat32Array());
        let maximum = 0;
        for (const value of values) maximum = Math.max(maximum, value);
        respond(
          {
            type: "mask",
            frameId: message.frameId,
            benchmarkId: message.benchmarkId,
            values: values.buffer,
            width: mask.width,
            height: mask.height,
            inferenceMs: performance.now() - startedAt,
            unreadable: result.landmarks.length > 0 && maximum <= 0.0001,
          },
          [values.buffer],
        );
      });
    }
  } catch (error) {
    respond({ type: "error", message: error instanceof Error ? error.message : "Worker inference failed" });
  } finally {
    message.bitmap.close();
  }
};

globalThis.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
  if (event.data.type === "init") {
    void initialize(event.data).catch((error) => {
      respond({ type: "error", message: error instanceof Error ? error.message : "Worker model initialization failed" });
    });
    return;
  }
  if (event.data.type === "reset") {
    requestRecurrentReset();
    return;
  }
  if (activeTask === "rvm") {
    void processRvmFrame(event.data).catch((error) => {
      respond({ type: "error", message: error instanceof Error ? error.message : "RVM inference failed" });
    });
    return;
  }
  processMediaPipeFrame(event.data);
});

export {};
