// AssemblyScript entry point — re-exports WASM inference API.
export {
  init,
  getWeightsPtr,
  getFeaturesPtr,
  infer,
  getResultIndex,
  getResultScore,
} from './inference';
