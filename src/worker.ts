/**
 * Web Worker that handles model loading, preprocessing, and inference.
 * All heavy computation happens here, keeping the main thread responsive.
 */

import { parseHzModel } from './model-loader';
import { parseWeights, runInference } from './inference';
import { preprocessStrokes } from './preprocessing';
import type { WorkerRequest, WorkerResponse, HanScribeResult } from './types';

type ModelWeights = ReturnType<typeof parseWeights>;

let weights: ModelWeights | null = null;

async function handleInit(modelUrl: string): Promise<void> {
  const resp = await fetch(modelUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch model: ${resp.status} ${resp.statusText}`);
  }
  const buffer = await resp.arrayBuffer();
  const model = parseHzModel(buffer);
  weights = parseWeights(model);
}

function handleRecognize(strokes: number[][][], topK: number): HanScribeResult[] {
  if (!weights) {
    throw new Error('Model not loaded');
  }

  const { data, numSegments } = preprocessStrokes(strokes);
  if (numSegments === 0) {
    return [];
  }

  const { indices, scores } = runInference(weights, data, numSegments, topK);

  return indices.map((idx, i) => ({
    char: weights!.vocab[idx],
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
