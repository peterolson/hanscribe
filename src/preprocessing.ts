/**
 * Stroke preprocessing: normalize, fit Bezier curves, extract 10-dim features.
 *
 * Feature layout per Bezier segment:
 *   [0] pen_down  — 1.0 for real strokes, 0.0 for pen-up segments
 *   [1] dx        — chord x-component in height=1 space
 *   [2] dy        — chord y-component in height=1 space
 *   [3] theta1    — angle of cp1 relative to chord
 *   [4] ratio1    — |cp1 offset| / chord_length
 *   [5] theta2    — angle of cp2 relative to reverse chord
 *   [6] ratio2    — |cp2 offset| / chord_length
 *   [7-9] 0       — time coefficients (unused)
 */

import { fitCurve } from './fit-curves';

/** Bezier fitting error tolerance. */
const FIT_ERROR = 0.2;

/** Default normalization scale. */
const DEFAULT_SCALE = 7;

type Point = [number, number];

/**
 * Normalize strokes: translate bounding box min to origin,
 * scale isometrically based on height.
 */
function normalizeStrokes(
  strokes: number[][][],
  scale: number,
): number[][][] {
  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;

  for (const stroke of strokes) {
    for (const [x, y] of stroke) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  const extent = Math.max(maxX - minX, maxY - minY);
  const s = extent > 0 ? scale / extent : 1;

  return strokes.map(stroke =>
    stroke.map(([x, y, t]) => [(x - minX) * s, (y - minY) * s, t])
  );
}

/** Compute 10-dim feature vector for a single cubic Bezier segment. */
function bezierFeatures(
  seg: Point[],
  penDown: boolean,
  scale: number,
): Float32Array {
  const [p0, p1, p2, p3] = seg;
  const f = new Float32Array(10);

  const dx = p3[0] - p0[0];
  const dy = p3[1] - p0[1];
  const chordLen = Math.sqrt(dx * dx + dy * dy);

  const d1x = p1[0] - p0[0];
  const d1y = p1[1] - p0[1];
  const d2x = p2[0] - p3[0];
  const d2y = p2[1] - p3[1];

  f[0] = penDown ? 1.0 : 0.0;
  f[1] = dx / scale;
  f[2] = dy / scale;

  const cross1 = dx * d1y - dy * d1x;
  const dot1 = dx * d1x + dy * d1y;
  f[3] = Math.atan2(cross1, dot1);

  if (chordLen > 0) {
    f[4] = Math.sqrt(d1x * d1x + d1y * d1y) / chordLen;
  }

  const cross2 = dy * d2x - dx * d2y;
  const dot2 = -(dx * d2x + dy * d2y);
  f[5] = Math.atan2(cross2, dot2);

  if (chordLen > 0) {
    f[6] = Math.sqrt(d2x * d2x + d2y * d2y) / chordLen;
  }

  return f;
}

/**
 * Convert raw strokes to a flat Float32Array of 10-dim Bezier features.
 *
 * @param strokes Array of strokes, each stroke is an array of [x, y, t] points
 * @param scale Normalization scale (default: 7)
 * @returns Features as Float32Array [T * 10] and segment count T
 */
export function preprocessStrokes(
  strokes: number[][][],
  scale: number = DEFAULT_SCALE,
): { data: Float32Array; numSegments: number } {
  const normalized = normalizeStrokes(strokes, scale);
  const allFeatures: Float32Array[] = [];

  for (let strokeIdx = 0; strokeIdx < normalized.length; strokeIdx++) {
    const stroke = normalized[strokeIdx];
    const points: Point[] = stroke.map(([x, y]) => [x, y]);

    if (points.length < 2) continue;

    // Insert pen-up segment between strokes
    if (strokeIdx > 0) {
      const prevStroke = normalized[strokeIdx - 1];
      const prevEnd: Point = [prevStroke[prevStroke.length - 1][0], prevStroke[prevStroke.length - 1][1]];
      const curStart: Point = [stroke[0][0], stroke[0][1]];
      // 1/3 control points (matches ML Kit)
      const penUpP1: Point = [
        prevEnd[0] + (curStart[0] - prevEnd[0]) / 3,
        prevEnd[1] + (curStart[1] - prevEnd[1]) / 3,
      ];
      const penUpP2: Point = [
        curStart[0] - (curStart[0] - prevEnd[0]) / 3,
        curStart[1] - (curStart[1] - prevEnd[1]) / 3,
      ];
      allFeatures.push(bezierFeatures([prevEnd, penUpP1, penUpP2, curStart], false, scale));
    }

    // Fit Bezier curves
    let beziers: Point[][];
    try {
      beziers = fitCurve(points, FIT_ERROR);
    } catch {
      beziers = [[points[0], points[0], points[points.length - 1], points[points.length - 1]]];
    }

    if (beziers.length === 0) continue;

    for (const seg of beziers) {
      allFeatures.push(bezierFeatures(seg, true, scale));
    }
  }

  const numSegments = allFeatures.length;
  const data = new Float32Array(numSegments * 10);
  for (let i = 0; i < numSegments; i++) {
    data.set(allFeatures[i], i * 10);
  }

  return { data, numSegments };
}
