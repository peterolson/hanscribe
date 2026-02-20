/**
 * Schneider's algorithm for fitting cubic Bezier curves to a sequence of points.
 * Original: Philip J. Schneider, "Graphics Gems", Academic Press, 1990.
 */

type Point = [number, number];
type BezierCurve = [Point, Point, Point, Point];

function add(a: Point, b: Point): Point {
  return [a[0] + b[0], a[1] + b[1]];
}

function sub(a: Point, b: Point): Point {
  return [a[0] - b[0], a[1] - b[1]];
}

function scale(v: Point, s: number): Point {
  return [v[0] * s, v[1] * s];
}

function dot(a: Point, b: Point): number {
  return a[0] * b[0] + a[1] * b[1];
}

function norm(v: Point): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
}

function normalize(v: Point): Point {
  const n = norm(v);
  return n === 0 ? [0, 0] : [v[0] / n, v[1] / n];
}

function q(cp: BezierCurve, t: number): Point {
  const mt = 1 - t;
  return [
    mt * mt * mt * cp[0][0] + 3 * mt * mt * t * cp[1][0] + 3 * mt * t * t * cp[2][0] + t * t * t * cp[3][0],
    mt * mt * mt * cp[0][1] + 3 * mt * mt * t * cp[1][1] + 3 * mt * t * t * cp[2][1] + t * t * t * cp[3][1],
  ];
}

function qprime(cp: BezierCurve, t: number): Point {
  const mt = 1 - t;
  return [
    3 * mt * mt * (cp[1][0] - cp[0][0]) + 6 * mt * t * (cp[2][0] - cp[1][0]) + 3 * t * t * (cp[3][0] - cp[2][0]),
    3 * mt * mt * (cp[1][1] - cp[0][1]) + 6 * mt * t * (cp[2][1] - cp[1][1]) + 3 * t * t * (cp[3][1] - cp[2][1]),
  ];
}

function qprimeprime(cp: BezierCurve, t: number): Point {
  const mt = 1 - t;
  return [
    6 * mt * (cp[2][0] - 2 * cp[1][0] + cp[0][0]) + 6 * t * (cp[3][0] - 2 * cp[2][0] + cp[1][0]),
    6 * mt * (cp[2][1] - 2 * cp[1][1] + cp[0][1]) + 6 * t * (cp[3][1] - 2 * cp[2][1] + cp[1][1]),
  ];
}

function chordLengthParameterize(points: Point[]): number[] {
  const u = [0];
  for (let i = 1; i < points.length; i++) {
    u.push(u[i - 1] + norm(sub(points[i], points[i - 1])));
  }
  const total = u[u.length - 1];
  if (total > 0) {
    for (let i = 0; i < u.length; i++) u[i] /= total;
  }
  return u;
}

function generateBezier(
  points: Point[],
  parameters: number[],
  leftTangent: Point,
  rightTangent: Point,
): BezierCurve {
  const bezCurve: BezierCurve = [points[0], [0, 0], [0, 0], points[points.length - 1]];

  const A: [Point, Point][] = [];
  for (const u of parameters) {
    A.push([
      scale(leftTangent, 3 * (1 - u) * (1 - u) * u),
      scale(rightTangent, 3 * (1 - u) * u * u),
    ]);
  }

  let C00 = 0, C01 = 0, C11 = 0, X0 = 0, X1 = 0;
  for (let i = 0; i < points.length; i++) {
    C00 += dot(A[i][0], A[i][0]);
    C01 += dot(A[i][0], A[i][1]);
    C11 += dot(A[i][1], A[i][1]);

    const tmp = sub(
      points[i],
      q([points[0], points[0], points[points.length - 1], points[points.length - 1]], parameters[i]),
    );
    X0 += dot(A[i][0], tmp);
    X1 += dot(A[i][1], tmp);
  }

  const det_C0_C1 = C00 * C11 - C01 * C01;
  const det_X_C1 = X0 * C11 - X1 * C01;
  const det_C0_X = C00 * X1 - C01 * X0;

  const alpha_l = det_C0_C1 === 0 ? 0 : det_X_C1 / det_C0_C1;
  const alpha_r = det_C0_C1 === 0 ? 0 : det_C0_X / det_C0_C1;

  const segLength = norm(sub(points[0], points[points.length - 1]));
  const epsilon = 1.0e-6 * segLength;

  if (alpha_l < epsilon || alpha_r < epsilon) {
    bezCurve[1] = add(bezCurve[0], scale(leftTangent, segLength / 3));
    bezCurve[2] = add(bezCurve[3], scale(rightTangent, segLength / 3));
  } else {
    bezCurve[1] = add(bezCurve[0], scale(leftTangent, alpha_l));
    bezCurve[2] = add(bezCurve[3], scale(rightTangent, alpha_r));
  }

  return bezCurve;
}

function newtonRaphsonRootFind(bez: BezierCurve, point: Point, u: number): number {
  const d = sub(q(bez, u), point);
  const qp = qprime(bez, u);
  const qpp = qprimeprime(bez, u);
  const numerator = dot(d, qp);
  const denominator = dot(qp, qp) + dot(d, qpp);
  if (denominator === 0) return u;
  return u - numerator / denominator;
}

function reparameterize(bez: BezierCurve, points: Point[], parameters: number[]): number[] {
  return parameters.map((u, i) => newtonRaphsonRootFind(bez, points[i], u));
}

function computeMaxError(
  points: Point[],
  bez: BezierCurve,
  parameters: number[],
): [number, number] {
  let maxDist = 0;
  let splitPoint = Math.floor(points.length / 2);
  for (let i = 0; i < points.length; i++) {
    const diff = sub(q(bez, parameters[i]), points[i]);
    const dist = diff[0] * diff[0] + diff[1] * diff[1];
    if (dist > maxDist) {
      maxDist = dist;
      splitPoint = i;
    }
  }
  return [maxDist, splitPoint];
}

function fitCubic(
  points: Point[],
  leftTangent: Point,
  rightTangent: Point,
  error: number,
): BezierCurve[] {
  if (points.length === 2) {
    const dist = norm(sub(points[0], points[1])) / 3;
    return [[
      points[0],
      add(points[0], scale(leftTangent, dist)),
      add(points[1], scale(rightTangent, dist)),
      points[1],
    ]];
  }

  let u = chordLengthParameterize(points);
  let bezCurve = generateBezier(points, u, leftTangent, rightTangent);
  let [maxError, splitPoint] = computeMaxError(points, bezCurve, u);

  if (maxError < error) return [bezCurve];

  if (maxError < error * error) {
    for (let i = 0; i < 20; i++) {
      const uPrime = reparameterize(bezCurve, points, u);
      bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent);
      [maxError, splitPoint] = computeMaxError(points, bezCurve, uPrime);
      if (maxError < error) return [bezCurve];
      u = uPrime;
    }
  }

  const centerTangent = normalize(sub(points[splitPoint - 1], points[splitPoint + 1]));
  const left = fitCubic(points.slice(0, splitPoint + 1), leftTangent, centerTangent, error);
  const right = fitCubic(points.slice(splitPoint), scale(centerTangent, -1), rightTangent, error);
  return [...left, ...right];
}

export function fitCurve(points: Point[], maxError: number): BezierCurve[] {
  if (points.length < 2) return [];
  const leftTangent = normalize(sub(points[1], points[0]));
  const rightTangent = normalize(sub(points[points.length - 2], points[points.length - 1]));
  return fitCubic(points, leftTangent, rightTangent, maxError);
}
