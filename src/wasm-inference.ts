/**
 * Host-side bridge for the AssemblyScript WASM inference engine.
 *
 * Provides the same runInference interface as the TypeScript engine but
 * delegates computation to the compiled WASM module.
 */

interface WasmExports {
  memory: WebAssembly.Memory;
  init(numLayers: number, H: number, numClasses: number, numFeatures: number, weightsLen: number): void;
  getWeightsPtr(): number;
  getFeaturesPtr(): number;
  infer(T: number, topK: number): number;
  getResultIndex(i: number): number;
  getResultScore(i: number): number;
}

export interface WasmInferenceEngine {
  runInference(
    features: Float32Array,
    T: number,
    topK: number,
  ): { indices: number[]; scores: number[] };
}

/**
 * Create a WASM inference engine from compiled WASM bytes.
 * Initializes immediately: copies weights into WASM memory once.
 *
 * @param wasmBytes The compiled .wasm binary (ArrayBuffer or Uint8Array)
 * @param header Model header with { numLayers, hiddenSize, numClasses, numFeatures }
 * @param rawWeights Deobfuscated weight bytes from the .hzmodel file
 */
export async function createWasmEngine(
  wasmBytes: ArrayBuffer | Uint8Array,
  header: { numLayers: number; hiddenSize: number; numClasses: number; numFeatures: number },
  rawWeights: Uint8Array,
): Promise<WasmInferenceEngine> {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exports = instance.exports as unknown as WasmExports;

  // Initialize: compute memory layout, grow memory, precompute offsets
  exports.init(
    header.numLayers,
    header.hiddenSize,
    header.numClasses,
    header.numFeatures,
    rawWeights.length,
  );

  // Copy raw weight bytes into WASM memory (one-time)
  const wPtr = exports.getWeightsPtr();
  new Uint8Array(exports.memory.buffer, wPtr, rawWeights.length).set(rawWeights);

  return {
    runInference(features, T, topK) {
      // Copy features into WASM memory
      const fPtr = exports.getFeaturesPtr();
      new Float32Array(exports.memory.buffer, fPtr, T * header.numFeatures)
        .set(features.subarray(0, T * header.numFeatures));

      // Run inference
      const count = exports.infer(T, topK);

      // Read results
      const indices: number[] = [];
      const scores: number[] = [];
      for (let i = 0; i < count; i++) {
        indices.push(exports.getResultIndex(i));
        scores.push(exports.getResultScore(i));
      }

      return { indices, scores };
    },
  };
}
