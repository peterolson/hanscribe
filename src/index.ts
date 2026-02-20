/**
 * HanScribe — Chinese handwriting recognition for the web.
 *
 * @example
 * ```js
 * const pad = await HanScribe.create({
 *   element: document.getElementById('draw-area'),
 *   onRecognize: (results) => console.log(results[0].char),
 * });
 * ```
 */

import { createCanvas } from './canvas';
// @ts-ignore — virtual module resolved by rollup plugin
import workerCode from 'worker-inline';
import type {
  HanScribeOptions,
  HanScribeResult,
  HanScribeInstance,
  WorkerRequest,
  WorkerResponse,
} from './types';

export type { HanScribeOptions, HanScribeResult, HanScribeInstance };

// Re-export utilities for advanced usage
export { parseHzModel } from './model-loader';
export { preprocessStrokes } from './preprocessing';
export { createWasmEngine } from './wasm-inference';
export type { WasmInferenceEngine } from './wasm-inference';

const DEFAULT_MODEL_URL =
  'https://github.com/peterolson/hanscribe/releases/latest/download/hanscribe.hzmodel';

/** Create an inline Web Worker from bundled source code. */
function spawnWorker(): Worker {
  const blob = new Blob([workerCode], { type: 'application/javascript' });
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url);
  // Revoke after worker has loaded the blob
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  return worker;
}

/** HanScribe static factory. */
export const HanScribe = {
  /**
   * Create a new HanScribe handwriting recognition pad.
   *
   * Spawns a Web Worker for inference so the main thread stays responsive.
   * Fetches and parses the model in the worker, sets up a drawing canvas,
   * and wires auto-recognition on stroke end (debounced).
   *
   * @param options Configuration options
   * @returns A promise that resolves when the model is loaded and the pad is ready
   */
  async create(options: HanScribeOptions): Promise<HanScribeInstance> {
    const {
      element,
      onRecognize,
      modelUrl = DEFAULT_MODEL_URL,
      strokeColor = '#000',
      strokeWidth = 3,
      topK = 10,
      debounceMs = 300,
    } = options;

    let debounceTimer: ReturnType<typeof setTimeout> | null = null;
    let destroyed = false;
    let recognizeId = 0;
    let pendingResolvers: Array<{
      id: number;
      resolve: (results: HanScribeResult[]) => void;
      reject: (err: Error) => void;
    }> = [];

    // Spawn worker and wire message handling
    const worker = spawnWorker();

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'result') {
        const resolver = pendingResolvers.shift();
        if (resolver) resolver.resolve(msg.results);
      } else if (msg.type === 'error') {
        const resolver = pendingResolvers.shift();
        if (resolver) resolver.reject(new Error(msg.message));
      }
    };

    worker.onerror = (e) => {
      const resolver = pendingResolvers.shift();
      if (resolver) resolver.reject(new Error(e.message));
    };

    function sendRecognize(strokes: number[][][]): Promise<HanScribeResult[]> {
      return new Promise((resolve, reject) => {
        const id = ++recognizeId;
        pendingResolvers.push({ id, resolve, reject });
        const msg: WorkerRequest = { type: 'recognize', strokes, topK };
        worker.postMessage(msg);
      });
    }

    // Create canvas with auto-recognize on stroke end
    const canvasCtrl = createCanvas(element, {
      strokeColor,
      strokeWidth,
      onStrokeEnd() {
        if (destroyed) return;
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          if (destroyed) return;
          const strokes = canvasCtrl.getStrokes();
          if (strokes.length === 0) return;
          sendRecognize(strokes).then(results => {
            if (onRecognize && !destroyed) onRecognize(results);
          }).catch(err => {
            console.error('HanScribe recognition error:', err);
          });
        }, debounceMs);
      },
    });

    // Initialize worker: send model URL, wait for 'ready'
    await new Promise<void>((resolve, reject) => {
      const initHandler = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data;
        if (msg.type === 'ready') {
          worker.removeEventListener('message', initHandler);
          resolve();
        } else if (msg.type === 'error') {
          worker.removeEventListener('message', initHandler);
          reject(new Error(msg.message));
        }
      };
      worker.addEventListener('message', initHandler);
      // Resolve to absolute URL since worker's base URL is a blob, not the page
      const absoluteModelUrl = new URL(modelUrl, location.href).href;
      const msg: WorkerRequest = { type: 'init', modelUrl: absoluteModelUrl };
      worker.postMessage(msg);
    });

    return {
      clear() {
        canvasCtrl.clear();
        if (debounceTimer) clearTimeout(debounceTimer);
      },

      destroy() {
        destroyed = true;
        if (debounceTimer) clearTimeout(debounceTimer);
        canvasCtrl.destroy();
        worker.terminate();
        pendingResolvers.forEach(r => r.reject(new Error('Destroyed')));
        pendingResolvers = [];
      },

      getStrokes() {
        return canvasCtrl.getStrokes();
      },

      recognize() {
        const strokes = canvasCtrl.getStrokes();
        if (strokes.length === 0) return Promise.resolve([]);
        return sendRecognize(strokes);
      },
    };
  },
};

export default HanScribe;
