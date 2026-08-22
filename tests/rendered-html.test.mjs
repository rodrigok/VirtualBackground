import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("ships a dark-first virtual background workbench", async () => {
  const [page, layout, styles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(layout, /VirtualBackground Lab/);
  assert.match(layout, /className="dark"/);
  assert.match(page, /Selfie Landscape/);
  assert.match(page, /Selfie Multiclass/);
  assert.match(page, /RVM MobileNetV3/);
  assert.match(page, /Internal downsample/);
  assert.match(page, /Predicted foreground/);
  assert.match(page, /RVM recurrent temporal state/);
  assert.match(page, /GPL-3\.0/);
  assert.match(page, /virtual-background-lab:settings:v1/);
  assert.match(page, /sanitizeSavedSettings/);
  assert.match(page, /window\.localStorage\.setItem/);
  assert.match(page, /window\.localStorage\.removeItem/);
  assert.match(page, /Modified · \$\{modifiedSettingCount\} saved/);
  assert.match(page, /Restore defaults and clear saved selections/);
  assert.match(page, /Tabs value=\{settingsTab\}/);
  assert.match(page, /useState<SegmentationModel>\("selfie-multiclass"\)/);
  assert.match(page, /Edge-aware refinement/);
  assert.match(page, /Guided edge upsample/);
  assert.match(page, /Other-class suppression/);
  assert.match(page, /Composite matte/);
  assert.match(page, /fillEnclosedMaskHoles/);
  assert.match(page, /guidedUpsampleMask/);
  assert.match(page, /Blur edge guard/);
  assert.match(page, /Replacement light wrap/);
  assert.match(page, /Hair \+ face recovery/);
  assert.match(page, /Skin boundary cleanup/);
  assert.match(page, /Edge color cleanup/);
  assert.match(page, /createEdgeCorrectionCanvas/);
  assert.match(page, /off on black/);
  assert.match(page, /Worker inference/);
  assert.match(page, /Frame-synced video/);
  assert.match(page, /Matte rate/);
  assert.match(page, /Motion edge cleanup/);
  assert.match(page, /Pose Heavy/);
  assert.match(page, /Input source/);
  assert.match(page, /4-up benchmark/);
  assert.match(page, /4 independent mattes/);
  assert.match(page, /benchmarkRawMasksRef/);
  assert.match(page, /Backgrounds/);
  assert.match(page, /Mask inference rate/);
  assert.match(page, /maskCanvas\.width !== compositeMaskWidth/);
  assert.match(page, /values\.length !== maskWidth \* maskHeight/);
  assert.match(page, /GPU mask unavailable · switching to CPU/);
  assert.match(page, /Frame health/);
  assert.ok(page.indexOf('id: "black"') < page.indexOf('id: "midnight"'));
  assert.match(page, /\[\.\.\.PROCEDURAL_BACKGROUNDS, \.\.\.IMAGE_BACKGROUNDS\]/);
  assert.doesNotMatch(page, /Violet grid/);
  assert.match(styles, /color-scheme:\s*dark/);
});

test("includes local processing assets", async () => {
  const [poseLite, poseFull, poseHeavy, selfieLandscape, selfieMulticlass, visionBundle, wasm, rvmModel, rvmWeights, rvmLicense, video, portrait, rightHand, leftHand, motionStill] = await Promise.all([
    readFile(new URL("../public/mediapipe/pose_landmarker_lite.task", import.meta.url)),
    readFile(new URL("../public/mediapipe/pose_landmarker_full.task", import.meta.url)),
    readFile(new URL("../public/mediapipe/pose_landmarker_heavy.task", import.meta.url)),
    readFile(new URL("../public/mediapipe/selfie_segmenter_landscape.tflite", import.meta.url)),
    readFile(new URL("../public/mediapipe/selfie_multiclass_256x256.tflite", import.meta.url)),
    readFile(new URL("../public/mediapipe/vision_bundle.mjs", import.meta.url)),
    readFile(new URL("../public/mediapipe/wasm/vision_wasm_internal.wasm", import.meta.url)),
    readFile(new URL("../public/rvm/model.json", import.meta.url)),
    readFile(new URL("../public/rvm/group1-shard1of1.bin", import.meta.url)),
    readFile(new URL("../public/rvm/LICENSE", import.meta.url)),
    readFile(new URL("../public/samples/user-motion.mov", import.meta.url)),
    readFile(new URL("../public/samples/user-photo-1.jpg", import.meta.url)),
    readFile(new URL("../public/samples/user-photo-2.jpg", import.meta.url)),
    readFile(new URL("../public/samples/user-photo-3.jpg", import.meta.url)),
    readFile(new URL("../public/samples/user-photo-4.jpg", import.meta.url)),
  ]);

  assert.ok(poseLite.byteLength > 1_000_000);
  assert.ok(poseFull.byteLength > poseLite.byteLength);
  assert.ok(poseHeavy.byteLength > poseFull.byteLength);
  assert.ok(selfieLandscape.byteLength > 200_000);
  assert.ok(selfieMulticlass.byteLength > 10_000_000);
  assert.ok(visionBundle.byteLength > 100_000);
  assert.ok(wasm.byteLength > 1_000_000);
  assert.ok(rvmModel.byteLength > 500_000);
  assert.ok(rvmWeights.byteLength > 3_000_000);
  assert.ok(rvmLicense.byteLength > 30_000);
  assert.ok(video.byteLength > 1_000_000);
  assert.ok(portrait.byteLength > 100_000);
  assert.ok(rightHand.byteLength > 100_000);
  assert.ok(leftHand.byteLength > 100_000);
  assert.ok(motionStill.byteLength > 100_000);

  const background = await readFile(new URL("../public/backgrounds/00-2026-AI-Gen-020.png", import.meta.url));
  assert.ok(background.byteLength > 100_000);
});
