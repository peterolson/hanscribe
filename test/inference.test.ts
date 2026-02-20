import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { parseHzModel } from '../src/model-loader';
import { parseWeights, runInference } from '../src/inference';
import { preprocessStrokes } from '../src/preprocessing';
import refData from './reference-data.json';

describe('parseWeights', () => {
  const modelBytes = readFileSync('test/test.hzmodel');
  const buffer = modelBytes.buffer.slice(
    modelBytes.byteOffset,
    modelBytes.byteOffset + modelBytes.byteLength,
  );
  const model = parseHzModel(buffer);

  it('parses 4 BiLSTM layers', () => {
    const weights = parseWeights(model);
    expect(weights.layers.length).toBe(4);
  });

  it('layer 1 has inputSize=10', () => {
    const weights = parseWeights(model);
    expect(weights.layers[0].inputSize).toBe(10);
  });

  it('layers 2-4 have inputSize=384', () => {
    const weights = parseWeights(model);
    for (let i = 1; i < 4; i++) {
      expect(weights.layers[i].inputSize).toBe(384);
    }
  });

  it('each direction has 4 input kernels and 4 rec kernels', () => {
    const weights = parseWeights(model);
    for (const layer of weights.layers) {
      for (const dir of [layer.forward, layer.backward]) {
        expect(dir.inputKernels.length).toBe(4);
        expect(dir.recKernels.length).toBe(4);
        expect(dir.biases.length).toBe(4);
        expect(dir.inputScales.length).toBe(4);
        expect(dir.recScales.length).toBe(4);
      }
    }
  });

  it('FC weights have correct dimensions', () => {
    const weights = parseWeights(model);
    expect(weights.fc.weights.length).toBe(12362 * 384);
    expect(weights.fc.biases.length).toBe(12362);
    expect(weights.fc.scale).toBeGreaterThan(0);
  });
});

describe('runInference', () => {
  const modelBytes = readFileSync('test/test.hzmodel');
  const buffer = modelBytes.buffer.slice(
    modelBytes.byteOffset,
    modelBytes.byteOffset + modelBytes.byteLength,
  );
  const model = parseHzModel(buffer);
  const weights = parseWeights(model);

  it('recognizes 好 as top-1 from reference strokes', () => {
    const { data, numSegments } = preprocessStrokes(refData.strokes);
    const { indices, scores } = runInference(weights, data, numSegments, 10);

    // The top character should be 好
    expect(indices.length).toBeGreaterThan(0);
    const topChar = weights.vocab[indices[0]];
    expect(topChar).toBe('好');

    // Score should be high (> 50%)
    expect(scores[0]).toBeGreaterThan(0.5);
  }, 30000); // 30s timeout for inference
});
