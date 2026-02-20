/**
 * BiLSTM + FC inference engine.
 *
 * Runs a 4-layer bidirectional LSTM followed by a fully-connected layer.
 * Weights are uint8-quantized (symmetric, zero_point=0, signed interpretation).
 * All computation is float32.
 */

import type { HzModel, HzModelHeader } from './types';

/** Parsed weight set for one LSTM direction (forward or backward). */
interface LSTMDirectionWeights {
  // Input kernels [hidden, input_size] + scale per gate (i, f, c, o)
  inputKernels: Uint8Array[];
  inputScales: number[];
  // Recurrent kernels [hidden, hidden] + scale per gate
  recKernels: Uint8Array[];
  recScales: number[];
  // Biases [hidden] per gate (float32)
  biases: Float32Array[];
}

interface BiLSTMLayerWeights {
  forward: LSTMDirectionWeights;
  backward: LSTMDirectionWeights;
  inputSize: number;
}

interface FCWeights {
  weights: Uint8Array;   // [numClasses, inputSize]
  scale: number;
  biases: Float32Array;  // [numClasses]
}

interface ModelWeights {
  layers: BiLSTMLayerWeights[];
  fc: FCWeights;
  header: HzModelHeader;
  vocab: string[];
}

/** Sigmoid activation. */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/** Tanh activation. */
function tanh(x: number): number {
  return Math.tanh(x);
}

/**
 * Matrix-vector multiply with dequantization: y += scale * W_int8 * x
 * W is stored as uint8 but interpreted as int8 (symmetric quantization).
 * W has shape [rows, cols], stored row-major.
 */
function matvecAddQuantized(
  y: Float32Array,
  w: Uint8Array,
  scale: number,
  x: Float32Array,
  rows: number,
  cols: number,
): void {
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    const rowOffset = i * cols;
    for (let j = 0; j < cols; j++) {
      // Reinterpret uint8 as int8
      let wVal = w[rowOffset + j];
      if (wVal > 127) wVal -= 256;
      sum += wVal * x[j];
    }
    y[i] += scale * sum;
  }
}

/** Parse weight data from deobfuscated weight bytes. */
export function parseWeights(model: HzModel): ModelWeights {
  const { header, vocab, weights } = model;
  const { numLayers, hiddenSize: H, numClasses, numFeatures } = header;

  let offset = 0;

  function readF32(): number {
    const view = new DataView(weights.buffer, weights.byteOffset + offset, 4);
    const val = view.getFloat32(0, true);
    offset += 4;
    return val;
  }

  function readUint8(n: number): Uint8Array {
    const data = weights.slice(offset, offset + n);
    offset += n;
    return data;
  }

  function readFloat32Array(n: number): Float32Array {
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = readF32();
    }
    return data;
  }

  const layers: BiLSTMLayerWeights[] = [];

  for (let l = 0; l < numLayers; l++) {
    const inputSize = l === 0 ? numFeatures : H * 2;

    const directions: LSTMDirectionWeights[] = [];
    for (let d = 0; d < 2; d++) {
      const inputKernels: Uint8Array[] = [];
      const inputScales: number[] = [];
      // 4 gates: input, cell, forget, output (matches converter order)
      for (let g = 0; g < 4; g++) {
        inputScales.push(readF32());
        inputKernels.push(readUint8(H * inputSize));
      }

      const recKernels: Uint8Array[] = [];
      const recScales: number[] = [];
      for (let g = 0; g < 4; g++) {
        recScales.push(readF32());
        recKernels.push(readUint8(H * H));
      }

      const biases: Float32Array[] = [];
      for (let g = 0; g < 4; g++) {
        biases.push(readFloat32Array(H));
      }

      directions.push({ inputKernels, inputScales, recKernels, recScales, biases });
    }

    layers.push({ forward: directions[0], backward: directions[1], inputSize });
  }

  // FC layer
  const fcScale = readF32();
  const fcWeights = readUint8(numClasses * H * 2);
  const fcBiases = readFloat32Array(numClasses);

  return {
    layers,
    fc: { weights: fcWeights, scale: fcScale, biases: fcBiases },
    header,
    vocab,
  };
}

/**
 * Run one direction of LSTM for all timesteps.
 * Returns output[T][H] as a flat Float32Array.
 */
function runLSTMDirection(
  dir: LSTMDirectionWeights,
  input: Float32Array,  // [T * inputSize]
  T: number,
  inputSize: number,
  H: number,
  reverse: boolean,
): Float32Array {
  const output = new Float32Array(T * H);
  const h = new Float32Array(H);
  const c = new Float32Array(H);
  // Gate order matches converter: [input, cell, forget, output]
  const gates = new Float32Array(4 * H);

  for (let step = 0; step < T; step++) {
    const t = reverse ? T - 1 - step : step;
    const xOffset = t * inputSize;

    // Reset gates to biases
    for (let g = 0; g < 4; g++) {
      const gateOffset = g * H;
      for (let i = 0; i < H; i++) {
        gates[gateOffset + i] = dir.biases[g][i];
      }
    }

    // Add input contribution: gates += W_input * x
    const xSlice = input.subarray(xOffset, xOffset + inputSize);
    for (let g = 0; g < 4; g++) {
      const gateSlice = new Float32Array(gates.buffer, g * H * 4, H);
      matvecAddQuantized(gateSlice, dir.inputKernels[g], dir.inputScales[g], xSlice, H, inputSize);
    }

    // Add recurrent contribution: gates += W_rec * h_prev
    for (let g = 0; g < 4; g++) {
      const gateSlice = new Float32Array(gates.buffer, g * H * 4, H);
      matvecAddQuantized(gateSlice, dir.recKernels[g], dir.recScales[g], h, H, H);
    }

    // Apply activations and update state
    // Gate layout: [0]=input, [1]=cell, [2]=forget, [3]=output
    for (let i = 0; i < H; i++) {
      const ig = sigmoid(gates[i]);           // input gate
      const cg = tanh(gates[H + i]);          // cell candidate
      const fg = sigmoid(gates[2 * H + i]);   // forget gate
      const og = sigmoid(gates[3 * H + i]);   // output gate

      c[i] = fg * c[i] + ig * cg;
      h[i] = og * tanh(c[i]);
    }

    // Store output
    output.set(h, t * H);
  }

  return output;
}

/**
 * Run one BiLSTM layer.
 * Input: [T, inputSize], Output: [T, 2*H]
 */
function runBiLSTMLayer(
  layer: BiLSTMLayerWeights,
  input: Float32Array,
  T: number,
  H: number,
): Float32Array {
  const fwOut = runLSTMDirection(layer.forward, input, T, layer.inputSize, H, false);
  const bwOut = runLSTMDirection(layer.backward, input, T, layer.inputSize, H, true);

  // Concatenate: output[t] = [fw[t], bw[t]]
  const output = new Float32Array(T * H * 2);
  for (let t = 0; t < T; t++) {
    const outOffset = t * H * 2;
    output.set(fwOut.subarray(t * H, (t + 1) * H), outOffset);
    output.set(bwOut.subarray(t * H, (t + 1) * H), outOffset + H);
  }

  return output;
}

/**
 * Run the FC layer and return softmax probabilities.
 * Input: [T, 2*H], Output: max probability per class across all timesteps.
 */
function runFCAndDecode(
  fc: FCWeights,
  input: Float32Array,
  T: number,
  inputSize: number,
  numClasses: number,
  topK: number,
): { indices: number[]; scores: number[] } {
  // Track max probability per class across all timesteps
  const maxProb = new Float32Array(numClasses);

  for (let t = 0; t < T; t++) {
    const x = input.subarray(t * inputSize, (t + 1) * inputSize);

    // Compute logits = W * x + b
    const logits = new Float32Array(numClasses);
    for (let i = 0; i < numClasses; i++) {
      let sum = 0;
      const rowOffset = i * inputSize;
      for (let j = 0; j < inputSize; j++) {
        let wVal = fc.weights[rowOffset + j];
        if (wVal > 127) wVal -= 256;
        sum += wVal * x[j];
      }
      logits[i] = fc.scale * sum + fc.biases[i];
    }

    // Softmax
    let maxLogit = -Infinity;
    for (let i = 0; i < numClasses; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    let sumExp = 0;
    for (let i = 0; i < numClasses; i++) {
      logits[i] = Math.exp(logits[i] - maxLogit);
      sumExp += logits[i];
    }
    for (let i = 0; i < numClasses; i++) {
      logits[i] /= sumExp;
    }

    // Update max prob (skip blank token at index numClasses - 1)
    for (let i = 0; i < numClasses - 1; i++) {
      if (logits[i] > maxProb[i]) {
        maxProb[i] = logits[i];
      }
    }
  }

  // Find top-K
  const indexed: [number, number][] = [];
  for (let i = 0; i < numClasses - 1; i++) {
    if (maxProb[i] > 0.001) {
      indexed.push([i, maxProb[i]]);
    }
  }
  indexed.sort((a, b) => b[1] - a[1]);
  const topResults = indexed.slice(0, topK);

  return {
    indices: topResults.map(r => r[0]),
    scores: topResults.map(r => r[1]),
  };
}

/**
 * Run full inference: 4 BiLSTM layers + FC + decode.
 *
 * @param weights Parsed model weights
 * @param features Float32Array of shape [T, numFeatures] (flattened)
 * @param T Number of timesteps (Bezier segments)
 * @param topK Number of top candidates to return
 * @returns Top-K character indices and scores
 */
export function runInference(
  weights: ModelWeights,
  features: Float32Array,
  T: number,
  topK: number,
): { indices: number[]; scores: number[] } {
  const H = weights.header.hiddenSize;

  // Run 4 BiLSTM layers
  let current = features;
  for (const layer of weights.layers) {
    current = runBiLSTMLayer(layer, current, T, H);
  }

  // FC + softmax + decode
  return runFCAndDecode(
    weights.fc,
    current,
    T,
    H * 2,
    weights.header.numClasses,
    topK,
  );
}
