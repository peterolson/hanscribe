/** Sigmoid activation (f32). */
@inline export function sigmoid(x: f32): f32 {
  return 1.0 / (1.0 + Mathf.exp(-x));
}

/** Tanh activation (f32). */
@inline export function tanhf(x: f32): f32 {
  const e2x: f32 = Mathf.exp(2.0 * x);
  return (e2x - 1.0) / (e2x + 1.0);
}
