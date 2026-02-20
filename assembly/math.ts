// AssemblyScript math helpers — placeholder for future WASM optimization.
// Currently, inference runs in TypeScript (src/inference.ts).
// This file will be implemented when porting hot loops to WASM for SIMD.

export function sigmoid(x: f32): f32 {
  if (x >= 0) {
    return 1.0 / (1.0 + <f32>Math.exp(<f64>(-x)));
  }
  const ex = <f32>Math.exp(<f64>x);
  return ex / (1.0 + ex);
}

export function tanhf(x: f32): f32 {
  return <f32>Math.tanh(<f64>x);
}
