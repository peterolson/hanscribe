/** Options for creating a HanScribe instance. */
export interface HanScribeOptions {
  /** Container element — the library creates a canvas inside it. */
  element: HTMLElement;
  /** Called with recognition results after each stroke (debounced). */
  onRecognize?: (results: HanScribeResult[]) => void;
  /** URL to the .hzmodel file. Defaults to jsdelivr CDN. */
  modelUrl?: string;
  /** Stroke color. Default: '#000' */
  strokeColor?: string;
  /** Stroke width in CSS pixels. Default: 3 */
  strokeWidth?: number;
  /** Number of top candidates to return. Default: 10 */
  topK?: number;
  /** Debounce delay in ms after stroke end before auto-recognizing. Default: 300 */
  debounceMs?: number;
}

/** A single recognition result. */
export interface HanScribeResult {
  /** The recognized character. */
  char: string;
  /** Probability score (0–1). */
  score: number;
}

/** The public API for a HanScribe drawing pad. */
export interface HanScribeInstance {
  /** Clear all strokes and results. */
  clear(): void;
  /** Destroy the instance: terminate worker, remove event listeners. */
  destroy(): void;
  /** Get current strokes as [x, y, t][][] */
  getStrokes(): number[][][];
  /** Trigger recognition manually. Returns top-K results. */
  recognize(): Promise<HanScribeResult[]>;
}

/** Messages sent from main thread → worker. */
export type WorkerRequest =
  | { type: 'init'; modelUrl: string }
  | { type: 'recognize'; strokes: number[][][]; topK: number };

/** Messages sent from worker → main thread. */
export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'result'; results: HanScribeResult[] }
  | { type: 'error'; message: string };

/** Parsed .hzmodel header. */
export interface HzModelHeader {
  version: number;
  flags: number;
  vocabOffset: number;
  vocabLength: number;
  weightsOffset: number;
  weightsLength: number;
  numLayers: number;
  hiddenSize: number;
  numClasses: number;
  numFeatures: number;
}

/** Parsed .hzmodel contents. */
export interface HzModel {
  header: HzModelHeader;
  vocab: string[];
  weights: Uint8Array;
}
