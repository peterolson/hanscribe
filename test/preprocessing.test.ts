import { describe, it, expect } from 'vitest';
import { preprocessStrokes } from '../src/preprocessing';
import refData from './reference-data.json';

describe('preprocessStrokes', () => {
  it('produces the correct number of segments for 好', () => {
    const { numSegments } = preprocessStrokes(refData.strokes);
    // Allow some tolerance since JS and Python fitCurve may differ slightly
    expect(numSegments).toBeGreaterThan(10);
    expect(numSegments).toBeLessThan(30);
  });

  it('first feature has pen_down=1 (real stroke)', () => {
    const { data } = preprocessStrokes(refData.strokes);
    expect(data[0]).toBeCloseTo(1.0, 3); // pen_down
  });

  it('pen-up segments have pen_down=0', () => {
    const { data, numSegments } = preprocessStrokes(refData.strokes);
    // Find a pen-up segment (pen_down=0)
    let foundPenUp = false;
    for (let i = 0; i < numSegments; i++) {
      if (data[i * 10] === 0) {
        foundPenUp = true;
        // Pen-up segments should have ratio ≈ 0.333 (1/3 control points)
        expect(data[i * 10 + 4]).toBeCloseTo(0.3333, 2);
        expect(data[i * 10 + 6]).toBeCloseTo(0.3333, 2);
        break;
      }
    }
    // With 5 strokes, there should be 4 pen-up segments
    expect(foundPenUp).toBe(true);
  });

  it('dx/dy are in height=1 space (small values)', () => {
    const { data, numSegments } = preprocessStrokes(refData.strokes);
    for (let i = 0; i < numSegments; i++) {
      const dx = data[i * 10 + 1];
      const dy = data[i * 10 + 2];
      // With scale=7, dx/dy should be in range [-2, 2] approximately
      expect(Math.abs(dx)).toBeLessThan(3);
      expect(Math.abs(dy)).toBeLessThan(3);
    }
  });

  it('time features are zero', () => {
    const { data, numSegments } = preprocessStrokes(refData.strokes);
    for (let i = 0; i < numSegments; i++) {
      expect(data[i * 10 + 7]).toBe(0);
      expect(data[i * 10 + 8]).toBe(0);
      expect(data[i * 10 + 9]).toBe(0);
    }
  });

  it('ratios are non-negative', () => {
    const { data, numSegments } = preprocessStrokes(refData.strokes);
    for (let i = 0; i < numSegments; i++) {
      expect(data[i * 10 + 4]).toBeGreaterThanOrEqual(0);
      expect(data[i * 10 + 6]).toBeGreaterThanOrEqual(0);
    }
  });
});
