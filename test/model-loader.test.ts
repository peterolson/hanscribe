import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { parseHzModel } from '../src/model-loader';

describe('parseHzModel', () => {
  const modelBytes = readFileSync('test/test.hzmodel');
  const buffer = modelBytes.buffer.slice(
    modelBytes.byteOffset,
    modelBytes.byteOffset + modelBytes.byteLength,
  );

  it('parses header correctly', () => {
    const model = parseHzModel(buffer);
    expect(model.header.version).toBe(1);
    expect(model.header.numLayers).toBe(4);
    expect(model.header.hiddenSize).toBe(192);
    expect(model.header.numClasses).toBe(12362);
    expect(model.header.numFeatures).toBe(10);
  });

  it('parses vocab with correct length', () => {
    const model = parseHzModel(buffer);
    expect(model.vocab.length).toBe(12361);
  });

  it('vocab contains expected characters', () => {
    const model = parseHzModel(buffer);
    expect(model.vocab).toContain('好');
    expect(model.vocab).toContain('中');
    expect(model.vocab).toContain('人');
  });

  it('weights are deobfuscated (not all zeros)', () => {
    const model = parseHzModel(buffer);
    let nonZero = 0;
    for (let i = 0; i < Math.min(1000, model.weights.length); i++) {
      if (model.weights[i] !== 0) nonZero++;
    }
    expect(nonZero).toBeGreaterThan(0);
  });

  it('rejects invalid magic', () => {
    const badBuffer = new ArrayBuffer(40);
    new Uint8Array(badBuffer).fill(0);
    expect(() => parseHzModel(badBuffer)).toThrow('bad magic');
  });
});
