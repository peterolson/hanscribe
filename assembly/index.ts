// AssemblyScript entry point — placeholder for future WASM optimization.
// Currently, all inference runs in TypeScript via Web Worker.
//
// Future plan: port the hot matrix-vector multiply loop to WASM with SIMD
// for ~4x speedup on the BiLSTM and FC layers.

export function add(a: i32, b: i32): i32 {
  return a + b;
}
