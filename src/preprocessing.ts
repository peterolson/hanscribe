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
 *   [7] time_chord — T3-T0: chord length in normalized space
 *   [8] time_cp1   — T1-T0: first time control point offset (chord/3 for uniform speed)
 *   [9] time_cp2   — T2-T3: second time control point offset from end (-chord/3 for uniform)
 */

import { fitCurve } from './fit-curves';

/** Bezier fitting error tolerance. */
const FIT_ERROR = 0.04;

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

/**
 * Fit 1D time Bezier control points T1, T2 given timestamps and u parameters.
 * Uses least-squares approach matching the spatial Bezier parameterization.
 *
 * Given: T(u) = T0(1-u)³ + 3T1·u(1-u)² + 3T2·u²(1-u) + T3·u³
 * With T0, T3 known, solve for T1, T2 that best fit timestamps[i] at params[i].
 */
function fitTimeBezier(
  timestamps: number[],
  params: number[],
  T0: number,
  T3: number,
): [number, number] {
  let C00 = 0, C01 = 0, C11 = 0, X0 = 0, X1 = 0;
  for (let i = 0; i < timestamps.length; i++) {
    const u = params[i];
    const mu = 1 - u;
    const A0 = 3 * mu * mu * u;     // basis for T1
    const A1 = 3 * mu * u * u;       // basis for T2
    C00 += A0 * A0;
    C01 += A0 * A1;
    C11 += A1 * A1;
    const target = timestamps[i] - (mu * mu * mu * T0 + u * u * u * T3);
    X0 += A0 * target;
    X1 += A1 * target;
  }

  const det = C00 * C11 - C01 * C01;
  if (Math.abs(det) < 1e-12) {
    return [T0 + (T3 - T0) / 3, T3 - (T3 - T0) / 3];
  }

  const T1 = (X0 * C11 - X1 * C01) / det;
  const T2 = (C00 * X1 - C01 * X0) / det;
  return [T1, T2];
}

/** Chord-length parameterization for a sequence of 2D points. */
function chordLengthParams(points: Point[]): number[] {
  const u = [0];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i][0] - points[i - 1][0];
    const dy = points[i][1] - points[i - 1][1];
    u.push(u[i - 1] + Math.sqrt(dx * dx + dy * dy));
  }
  const total = u[u.length - 1];
  if (total > 0) for (let i = 0; i < u.length; i++) u[i] /= total;
  return u;
}

/** Compute 10-dim feature vector for a single cubic Bezier segment. */
function bezierFeatures(
  seg: Point[],
  penDown: boolean,
  scale: number,
  timeFeatures?: [number, number, number],
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

  // Time features: f[7]=T3-T0, f[8]=T1-T0, f[9]=T2-T3.
  if (timeFeatures) {
    f[7] = timeFeatures[0];
    f[8] = timeFeatures[1];
    f[9] = timeFeatures[2];
  } else {
    // Uniform-speed fallback: chord, chord/3, -chord/3
    const chordNorm = chordLen / scale;
    f[7] = chordNorm;
    f[8] = chordNorm / 3;
    f[9] = -chordNorm / 3;
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
  fitError: number = FIT_ERROR,
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
      // 1/3 control points
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
      beziers = fitCurve(points, fitError);
    } catch {
      beziers = [[points[0], points[0], points[points.length - 1], points[points.length - 1]]];
    }

    if (beziers.length === 0) continue;

    // Check if stroke has real timestamps (non-zero)
    const hasTimestamps = stroke[0][2] > 0 && stroke[stroke.length - 1][2] > 0;
    const timestamps = hasTimestamps ? stroke.map(p => p[2] / 1000) : null; // ms → seconds

    for (const seg of beziers) {
      let timeFeats: [number, number, number] | undefined;

      if (timestamps) {
        // Match segment endpoints to original points by reference equality
        const startIdx = points.indexOf(seg[0] as Point);
        const endIdx = points.indexOf(seg[3] as Point);

        if (startIdx >= 0 && endIdx > startIdx) {
          const segPoints = points.slice(startIdx, endIdx + 1);
          const segTimestamps = timestamps.slice(startIdx, endIdx + 1);
          const T0 = segTimestamps[0];
          const T3 = segTimestamps[segTimestamps.length - 1];

          if (segPoints.length >= 4) {
            const uParams = chordLengthParams(segPoints);
            const [T1, T2] = fitTimeBezier(segTimestamps, uParams, T0, T3);
            timeFeats = [T3 - T0, T1 - T0, T2 - T3];
          } else {
            // Too few points for meaningful fit, use uniform timing
            const duration = T3 - T0;
            timeFeats = [duration, duration / 3, -duration / 3];
          }
        }
      }

      allFeatures.push(bezierFeatures(seg, true, scale, timeFeats));
    }
  }

  const numSegments = allFeatures.length;
  const data = new Float32Array(numSegments * 10);
  for (let i = 0; i < numSegments; i++) {
    data.set(allFeatures[i], i * 10);
  }

  return { data, numSegments };
}
