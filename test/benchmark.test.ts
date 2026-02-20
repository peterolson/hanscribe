import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { parseHzModel } from '../src/model-loader';
import { parseWeights, runInference } from '../src/inference';
import { preprocessStrokes } from '../src/preprocessing';
import { createWasmEngine } from '../src/wasm-inference';
import refData from './reference-data.json';

describe('WASM inference benchmark', () => {
  const modelBytes = readFileSync('test/test.hzmodel');
  const buffer = modelBytes.buffer.slice(
    modelBytes.byteOffset,
    modelBytes.byteOffset + modelBytes.byteLength,
  );
  const model = parseHzModel(buffer);
  const wasmBytes = readFileSync('build/inference.wasm');

  const { data: features, numSegments: T } = preprocessStrokes(refData.strokes);
  const topK = 10;

  it('WASM produces correct top-1 result', async () => {
    const engine = await createWasmEngine(wasmBytes, model.header, model.weights);
    const result = engine.runInference(features, T, topK);

    expect(result.indices.length).toBeGreaterThan(0);
    const topChar = model.vocab[result.indices[0]];
    expect(topChar).toBe('好');
    expect(result.scores[0]).toBeGreaterThan(0.3);
  }, 60000);

  it('WASM matches TypeScript top-1', async () => {
    const tsWeights = parseWeights(model);
    const tsResult = runInference(tsWeights, features, T, topK);

    const engine = await createWasmEngine(wasmBytes, model.header, model.weights);
    const wasmResult = engine.runInference(features, T, topK);

    // Same top-1 character
    expect(wasmResult.indices[0]).toBe(tsResult.indices[0]);

    // Scores within tolerance (Mathf.exp vs Math.exp differences)
    expect(wasmResult.scores[0]).toBeCloseTo(tsResult.scores[0], 1);
  }, 60000);

  it('benchmark: TS vs WASM', async () => {
    const WARMUP = 3;
    const RUNS = 20;

    // --- TypeScript engine ---
    const tsWeights = parseWeights(model);

    // Warmup
    for (let i = 0; i < WARMUP; i++) {
      runInference(tsWeights, features, T, topK);
    }

    // Benchmark
    const tsTimes: number[] = [];
    for (let i = 0; i < RUNS; i++) {
      const start = performance.now();
      runInference(tsWeights, features, T, topK);
      tsTimes.push(performance.now() - start);
    }

    // --- WASM engine ---
    const engine = await createWasmEngine(wasmBytes, model.header, model.weights);

    // Warmup
    for (let i = 0; i < WARMUP; i++) {
      engine.runInference(features, T, topK);
    }

    // Benchmark
    const wasmTimes: number[] = [];
    for (let i = 0; i < RUNS; i++) {
      const start = performance.now();
      engine.runInference(features, T, topK);
      wasmTimes.push(performance.now() - start);
    }

    // Stats
    const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const stddev = (arr: number[]) => {
      const m = mean(arr);
      return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
    };

    const tsMean = mean(tsTimes);
    const tsStd = stddev(tsTimes);
    const wasmMean = mean(wasmTimes);
    const wasmStd = stddev(wasmTimes);
    const speedup = tsMean / wasmMean;

    console.log(`\n=== Inference Benchmark (T=${T}, topK=${topK}, ${RUNS} runs) ===`);
    console.log(`TypeScript:  ${tsMean.toFixed(1)} ± ${tsStd.toFixed(1)} ms`);
    console.log(`WASM:        ${wasmMean.toFixed(1)} ± ${wasmStd.toFixed(1)} ms`);
    console.log(`Speedup:     ${speedup.toFixed(2)}x`);

    // Just assert both produce results (benchmark is informational)
    expect(tsMean).toBeGreaterThan(0);
    expect(wasmMean).toBeGreaterThan(0);
  }, 300000); // 5 min timeout for benchmarking
});
