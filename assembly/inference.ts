/**
 * AssemblyScript BiLSTM + FC inference engine.
 *
 * Port of src/inference.ts. All buffers live at fixed precomputed offsets
 * in WASM linear memory — zero allocations during inference.
 *
 * Weight format matches the .hzmodel sequential layout:
 *   Per layer (4 total), per direction (fwd/bwd):
 *     4 gates × (scale:f32 + kernel:u8[H×inputSize])
 *     4 gates × (scale:f32 + kernel:u8[H×H])
 *     4 gates × (bias:f32[H])
 *   FC: scale:f32 + weights:u8[numClasses×2H] + biases:f32[numClasses]
 */

import { sigmoid, tanhf } from './math';

// ── Constants ───────────────────────────────────────────────────────────────

const MAX_T: i32 = 128;
const MAX_TOP_K: i32 = 100;
const NUM_GATES: i32 = 4;
const NUM_LAYERS: i32 = 4;

// ── Model parameters (set by init) ─────────────────────────────────────────

let _H: i32 = 0;
let _numClasses: i32 = 0;
let _numFeatures: i32 = 0;

// ── Memory layout offsets (computed by init) ────────────────────────────────

// Offset table: 8 direction base offsets + FC base + 4 inputSizes = 13 u32s
let _tableOff: u32 = 0;

let _weightsOff: u32 = 0;
let _featuresOff: u32 = 0;
let _outputAOff: u32 = 0;
let _outputBOff: u32 = 0;
let _tempFwOff: u32 = 0;
let _tempBwOff: u32 = 0;
let _gatesOff: u32 = 0;
let _hOff: u32 = 0;
let _cOff: u32 = 0;
let _logitsOff: u32 = 0;
let _maxProbOff: u32 = 0;
let _resultIdxOff: u32 = 0;
let _resultScrOff: u32 = 0;

// ── Offset table helpers ────────────────────────────────────────────────────

// Table layout (starting at _tableOff):
//   [0..7]  8 direction base offsets (u32) — index = layer*2 + dir
//   [8]     FC base offset (u32)
//   [9..12] 4 per-layer input sizes (u32)

@inline function storeDirBase(layer: i32, dir: i32, val: u32): void {
  store<u32>(_tableOff + u32((layer * 2 + dir) << 2), val);
}

@inline function loadDirBase(layer: i32, dir: i32): u32 {
  return load<u32>(_tableOff + u32((layer * 2 + dir) << 2));
}

@inline function storeFcBase(val: u32): void {
  store<u32>(_tableOff + 32, val);
}

@inline function loadFcBase(): u32 {
  return load<u32>(_tableOff + 32);
}

@inline function storeLayerInputSize(layer: i32, val: i32): void {
  store<u32>(_tableOff + 36 + u32(layer << 2), u32(val));
}

@inline function loadLayerInputSize(layer: i32): i32 {
  return i32(load<u32>(_tableOff + 36 + u32(layer << 2)));
}

// ── Exports ─────────────────────────────────────────────────────────────────

/**
 * Initialize the engine: compute memory layout, grow memory, precompute
 * weight offsets. Must be called once before any inference.
 */
export function init(
  numLayers: i32,
  H: i32,
  numClasses: i32,
  numFeatures: i32,
  weightsLen: i32,
): void {
  _H = H;
  _numClasses = numClasses;
  _numFeatures = numFeatures;

  const H2: i32 = H * 2;

  // Lay out memory regions starting from __heap_base
  _tableOff    = <u32>__heap_base;
  _weightsOff  = _tableOff + 64;
  _featuresOff = _weightsOff + u32(weightsLen);
  _outputAOff  = _featuresOff + u32(MAX_T * numFeatures) * 4;
  _outputBOff  = _outputAOff + u32(MAX_T * H2) * 4;
  _tempFwOff   = _outputBOff + u32(MAX_T * H2) * 4;
  _tempBwOff   = _tempFwOff  + u32(MAX_T * H) * 4;
  _gatesOff    = _tempBwOff  + u32(MAX_T * H) * 4;
  _hOff        = _gatesOff   + u32(NUM_GATES * H) * 4;
  _cOff        = _hOff       + u32(H) * 4;
  _logitsOff   = _cOff       + u32(H) * 4;
  _maxProbOff  = _logitsOff  + u32(numClasses) * 4;
  _resultIdxOff = _maxProbOff + u32(numClasses) * 4;
  _resultScrOff = _resultIdxOff + u32(MAX_TOP_K) * 4;

  const totalBytes: u32 = _resultScrOff + u32(MAX_TOP_K) * 4;

  // Grow memory if needed
  const currentPages: i32 = memory.size();
  const neededPages: i32 = i32((totalBytes + 0xFFFF) >> 16);
  if (neededPages > currentPages) {
    memory.grow(neededPages - currentPages);
  }

  // Precompute direction base offsets by walking the weight layout
  let wOff: u32 = _weightsOff;
  for (let l: i32 = 0; l < numLayers; l++) {
    const inputSize: i32 = l == 0 ? numFeatures : H2;
    storeLayerInputSize(l, inputSize);

    for (let d: i32 = 0; d < 2; d++) {
      storeDirBase(l, d, wOff);
      // Skip past: 4 input gates + 4 rec gates + 4 bias arrays
      wOff += u32(
        NUM_GATES * (4 + H * inputSize) +
        NUM_GATES * (4 + H * H) +
        NUM_GATES * H * 4
      );
    }
  }

  storeFcBase(wOff);
}

/** Pointer where the host should copy raw weight bytes. */
export function getWeightsPtr(): u32 {
  return _weightsOff;
}

/** Pointer where the host should write f32 features [T × numFeatures]. */
export function getFeaturesPtr(): u32 {
  return _featuresOff;
}

/**
 * Run full inference: 4 BiLSTM layers + FC + softmax + top-K decode.
 * Returns the number of results (≤ topK). Read results via
 * getResultIndex(i) / getResultScore(i).
 */
export function infer(T: i32, topK: i32): i32 {
  if (topK > MAX_TOP_K) topK = MAX_TOP_K;
  if (T > MAX_T) T = MAX_T;

  const H: i32 = _H;

  // Layer 0: input=features, output=A
  runBiLSTMLayer(0, _featuresOff, T, _outputAOff);
  // Layer 1: input=A, output=B
  runBiLSTMLayer(1, _outputAOff, T, _outputBOff);
  // Layer 2: input=B, output=A
  runBiLSTMLayer(2, _outputBOff, T, _outputAOff);
  // Layer 3: input=A, output=B
  runBiLSTMLayer(3, _outputAOff, T, _outputBOff);

  // FC + softmax + decode (input from B)
  return runFCAndDecode(_outputBOff, T, topK);
}

/** Get the i-th result's class index. */
export function getResultIndex(i: i32): i32 {
  return load<i32>(_resultIdxOff + u32(i) * 4);
}

/** Get the i-th result's score (probability). */
export function getResultScore(i: i32): f32 {
  return load<f32>(_resultScrOff + u32(i) * 4);
}

// ── Internal: matrix-vector multiply ────────────────────────────────────────

/**
 * y[i] += scale * sum_j(int8(W[i,j]) * x[j])
 *
 * W is uint8 in memory but interpreted as int8 (symmetric quantization).
 * W layout: row-major [rows × cols] at wOff.
 * y is f32[rows] at yOff, x is f32[cols] at xOff.
 */
function matvecAddQuantized(
  yOff: u32,
  wOff: u32,
  scale: f32,
  xOff: u32,
  rows: i32,
  cols: i32,
): void {
  for (let i: i32 = 0; i < rows; i++) {
    let sum: f32 = 0;
    const rowOff: u32 = wOff + u32(i * cols);
    for (let j: i32 = 0; j < cols; j++) {
      const wVal: f32 = f32(i32(load<i8>(rowOff + u32(j))));
      sum += wVal * load<f32>(xOff + u32(j << 2));
    }
    const yAddr: u32 = yOff + u32(i << 2);
    store<f32>(yAddr, load<f32>(yAddr) + scale * sum);
  }
}

// ── Internal: LSTM direction ────────────────────────────────────────────────

/**
 * Run one LSTM direction (forward or backward) for all timesteps.
 * Writes output[t][H] (f32) to outputOff.
 */
function runLSTMDirection(
  dirBase: u32,
  inputOff: u32,
  T: i32,
  inputSize: i32,
  H: i32,
  reverse: bool,
  outputOff: u32,
): void {
  // Zero h and c state vectors
  memory.fill(_hOff, 0, u32(H) << 2);
  memory.fill(_cOff, 0, u32(H) << 2);

  // Precompute strides for gate sub-offsets
  const inputGateStride: u32 = u32(4 + H * inputSize);
  const recGateStride: u32 = u32(4 + H * H);
  const biasBytes: u32 = u32(H) << 2;

  const recBaseOff: u32 = dirBase + u32(NUM_GATES) * inputGateStride;
  const biasBaseOff: u32 = recBaseOff + u32(NUM_GATES) * recGateStride;

  for (let step: i32 = 0; step < T; step++) {
    const t: i32 = reverse ? T - 1 - step : step;
    const xOff: u32 = inputOff + u32(t * inputSize) * 4;

    // Initialize gates with biases
    for (let g: i32 = 0; g < NUM_GATES; g++) {
      memory.copy(
        _gatesOff + u32(g * H) * 4,
        biasBaseOff + u32(g) * biasBytes,
        biasBytes,
      );
    }

    // gates += W_input * x (per gate)
    for (let g: i32 = 0; g < NUM_GATES; g++) {
      const scaleOff: u32 = dirBase + u32(g) * inputGateStride;
      const scale: f32 = load<f32>(scaleOff);
      const kernelOff: u32 = scaleOff + 4;
      matvecAddQuantized(
        _gatesOff + u32(g * H) * 4,
        kernelOff,
        scale,
        xOff,
        H,
        inputSize,
      );
    }

    // gates += W_rec * h_prev (per gate)
    for (let g: i32 = 0; g < NUM_GATES; g++) {
      const scaleOff: u32 = recBaseOff + u32(g) * recGateStride;
      const scale: f32 = load<f32>(scaleOff);
      const kernelOff: u32 = scaleOff + 4;
      matvecAddQuantized(
        _gatesOff + u32(g * H) * 4,
        kernelOff,
        scale,
        _hOff,
        H,
        H,
      );
    }

    // Apply activations and update cell/hidden state
    // Gate layout: [0]=input, [1]=cell, [2]=forget, [3]=output
    for (let i: i32 = 0; i < H; i++) {
      const iOff: u32 = u32(i) << 2;
      const ig: f32 = sigmoid(load<f32>(_gatesOff + iOff));
      const cg: f32 = tanhf(load<f32>(_gatesOff + u32(H) * 4 + iOff));
      const fg: f32 = sigmoid(load<f32>(_gatesOff + u32(2 * H) * 4 + iOff));
      const og: f32 = sigmoid(load<f32>(_gatesOff + u32(3 * H) * 4 + iOff));

      const newC: f32 = fg * load<f32>(_cOff + iOff) + ig * cg;
      store<f32>(_cOff + iOff, newC);

      const newH: f32 = og * tanhf(newC);
      store<f32>(_hOff + iOff, newH);
    }

    // Store h to output
    memory.copy(outputOff + u32(t * H) * 4, _hOff, u32(H) << 2);
  }
}

// ── Internal: BiLSTM layer ──────────────────────────────────────────────────

/**
 * Run one BiLSTM layer: forward + backward + concatenate.
 * Input: [T, inputSize] (f32) at inputOff
 * Output: [T, 2*H] (f32) at outputOff
 */
function runBiLSTMLayer(
  layer: i32,
  inputOff: u32,
  T: i32,
  outputOff: u32,
): void {
  const H: i32 = _H;
  const inputSize: i32 = loadLayerInputSize(layer);
  const hBytes: u32 = u32(H) << 2;

  // Forward → _tempFwOff [T, H]
  runLSTMDirection(
    loadDirBase(layer, 0),
    inputOff, T, inputSize, H, false,
    _tempFwOff,
  );

  // Backward → _tempBwOff [T, H]
  runLSTMDirection(
    loadDirBase(layer, 1),
    inputOff, T, inputSize, H, true,
    _tempBwOff,
  );

  // Interleave: output[t] = [fw[t], bw[t]]
  for (let t: i32 = 0; t < T; t++) {
    const outOff: u32 = outputOff + u32(t * H * 2) * 4;
    memory.copy(outOff, _tempFwOff + u32(t * H) * 4, hBytes);
    memory.copy(outOff + hBytes, _tempBwOff + u32(t * H) * 4, hBytes);
  }
}

// ── Internal: FC layer + decode ─────────────────────────────────────────────

/**
 * FC matmul + softmax + max-prob decode + top-K selection.
 * Returns number of results written.
 */
function runFCAndDecode(inputOff: u32, T: i32, topK: i32): i32 {
  const inputSize: i32 = _H * 2;
  const numClasses: i32 = _numClasses;

  // FC weight layout: scale(4) + weights(numClasses×inputSize) + biases(numClasses×4)
  const fcOff: u32 = loadFcBase();
  const fcScale: f32 = load<f32>(fcOff);
  const fcWeightsOff: u32 = fcOff + 4;
  const fcBiasesOff: u32 = fcWeightsOff + u32(numClasses * inputSize);
  const classBytesF32: u32 = u32(numClasses) << 2;

  // Zero maxProb
  memory.fill(_maxProbOff, 0, classBytesF32);

  for (let t: i32 = 0; t < T; t++) {
    const xOff: u32 = inputOff + u32(t * inputSize) * 4;

    // logits = biases (copy f32 values)
    memory.copy(_logitsOff, fcBiasesOff, classBytesF32);

    // logits += scale * W * x
    matvecAddQuantized(_logitsOff, fcWeightsOff, fcScale, xOff, numClasses, inputSize);

    // Softmax: find max logit
    let maxLogit: f32 = f32.MIN_VALUE;
    for (let i: i32 = 0; i < numClasses; i++) {
      const v: f32 = load<f32>(_logitsOff + u32(i << 2));
      if (v > maxLogit) maxLogit = v;
    }

    // exp and sum
    let sumExp: f32 = 0;
    for (let i: i32 = 0; i < numClasses; i++) {
      const addr: u32 = _logitsOff + u32(i << 2);
      const e: f32 = Mathf.exp(load<f32>(addr) - maxLogit);
      store<f32>(addr, e);
      sumExp += e;
    }

    // Update maxProb (skip blank at last index)
    const invSum: f32 = 1.0 / sumExp;
    for (let i: i32 = 0; i < numClasses - 1; i++) {
      const prob: f32 = load<f32>(_logitsOff + u32(i << 2)) * invSum;
      const mpAddr: u32 = _maxProbOff + u32(i << 2);
      if (prob > load<f32>(mpAddr)) {
        store<f32>(mpAddr, prob);
      }
    }
  }

  // Top-K selection
  return findTopK(topK);
}

// ── Internal: top-K selection ───────────────────────────────────────────────

/**
 * Find top-K classes by maxProb. Writes to _resultIdxOff / _resultScrOff.
 * Returns count of results (≤ topK).
 */
function findTopK(topK: i32): i32 {
  const numClasses: i32 = _numClasses;
  let count: i32 = 0;
  let minScore: f32 = 0;
  let minSlot: i32 = 0;

  for (let i: i32 = 0; i < numClasses - 1; i++) {
    const prob: f32 = load<f32>(_maxProbOff + u32(i << 2));
    if (prob <= 0.001) continue;

    if (count < topK) {
      store<i32>(_resultIdxOff + u32(count << 2), i);
      store<f32>(_resultScrOff + u32(count << 2), prob);
      if (count == 0 || prob < minScore) {
        minScore = prob;
        minSlot = count;
      }
      count++;
    } else if (prob > minScore) {
      // Replace the minimum entry
      store<i32>(_resultIdxOff + u32(minSlot << 2), i);
      store<f32>(_resultScrOff + u32(minSlot << 2), prob);
      // Re-scan for new minimum
      minScore = prob;
      minSlot = 0;
      for (let j: i32 = 0; j < topK; j++) {
        const s: f32 = load<f32>(_resultScrOff + u32(j << 2));
        if (s < minScore) {
          minScore = s;
          minSlot = j;
        }
      }
    }
  }

  // Insertion sort (descending) — K is small
  for (let i: i32 = 1; i < count; i++) {
    const idx: i32 = load<i32>(_resultIdxOff + u32(i << 2));
    const scr: f32 = load<f32>(_resultScrOff + u32(i << 2));
    let j: i32 = i - 1;
    while (j >= 0 && load<f32>(_resultScrOff + u32(j << 2)) < scr) {
      store<i32>(_resultIdxOff + u32((j + 1) << 2), load<i32>(_resultIdxOff + u32(j << 2)));
      store<f32>(_resultScrOff + u32((j + 1) << 2), load<f32>(_resultScrOff + u32(j << 2)));
      j--;
    }
    store<i32>(_resultIdxOff + u32((j + 1) << 2), idx);
    store<f32>(_resultScrOff + u32((j + 1) << 2), scr);
  }

  return count;
}
