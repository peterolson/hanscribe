/**
 * Web Worker that handles model loading, preprocessing, and inference.
 * All heavy computation happens here, keeping the main thread responsive.
 *
 * Inference runs in WASM (AssemblyScript) for ~4x speedup over TypeScript.
 */

import { parseHzModel } from './model-loader';
import { createWasmEngine, type WasmInferenceEngine } from './wasm-inference';
import { preprocessStrokes } from './preprocessing';
import type { WorkerRequest, WorkerResponse, HanScribeResult } from './types';

// @ts-ignore — virtual module resolved by rollup wasmInline plugin
import wasmBase64 from 'wasm-inline';

let engine: WasmInferenceEngine | null = null;
let vocab: string[] = [];

/** Decode base64 string to Uint8Array. */
function decodeBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

async function handleInit(modelUrl: string): Promise<void> {
  const resp = await fetch(modelUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch model: ${resp.status} ${resp.statusText}`);
  }
  const buffer = await resp.arrayBuffer();
  const model = parseHzModel(buffer);

  // Decode inlined WASM binary and create engine
  const wasmBytes = decodeBase64(wasmBase64);
  engine = await createWasmEngine(wasmBytes, model.header, model.weights);
  vocab = model.vocab;
}

function handleRecognize(strokes: number[][][], topK: number): HanScribeResult[] {
  if (!engine) {
    throw new Error('Model not loaded');
  }

  const { data, numSegments } = preprocessStrokes(strokes);
  if (numSegments === 0) {
    return [];
  }

  const { indices, scores } = engine.runInference(data, numSegments, topK);

  return indices.map((idx, i) => ({
    char: vocab[idx],
    score: scores[i],
  }));
}

function respond(msg: WorkerResponse) {
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;

  try {
    switch (msg.type) {
      case 'init':
        await handleInit(msg.modelUrl);
        respond({ type: 'ready' });
        break;

      case 'recognize': {
        const results = handleRecognize(msg.strokes, msg.topK);
        respond({ type: 'result', results });
        break;
      }
    }
  } catch (err) {
    respond({ type: 'error', message: String(err) });
  }
};
