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

  it('pen-up time features follow chord-length pattern', () => {
    const { data, numSegments } = preprocessStrokes(refData.strokes);
    for (let i = 0; i < numSegments; i++) {
      if (data[i * 10] > 0.5) continue; // skip pen-down
      const dx = data[i * 10 + 1];
      const dy = data[i * 10 + 2];
      const chordNorm = Math.sqrt(dx * dx + dy * dy);
      // Pen-up: f[7] = chord, f[8] = chord/3, f[9] = -chord/3
      expect(data[i * 10 + 7]).toBeCloseTo(chordNorm, 4);
      expect(data[i * 10 + 8]).toBeCloseTo(chordNorm / 3, 4);
      expect(data[i * 10 + 9]).toBeCloseTo(-chordNorm / 3, 4);
    }
  });

  it('pen-down time features are in reasonable range', () => {
    const { data, numSegments } = preprocessStrokes(refData.strokes);
    for (let i = 0; i < numSegments; i++) {
      if (data[i * 10] < 0.5) continue; // skip pen-up
      // f[7] = T3-T0 (duration, should be positive and < 5s)
      expect(data[i * 10 + 7]).toBeGreaterThanOrEqual(0);
      expect(data[i * 10 + 7]).toBeLessThan(5);
      // f[8] = T1-T0 (should be reasonable magnitude)
      expect(Math.abs(data[i * 10 + 8])).toBeLessThan(5);
      // f[9] = T2-T3 (typically negative)
      expect(Math.abs(data[i * 10 + 9])).toBeLessThan(5);
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
