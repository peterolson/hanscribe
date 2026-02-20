import type { HzModel, HzModelHeader } from './types';

const MAGIC = 0x44505A48; // "HZPD" as little-endian uint32
const XOR_KEY = [0x48, 0x5A, 0x50, 0x44]; // "HZPD"

function xorDeobfuscate(data: Uint8Array): Uint8Array {
  const result = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i] ^ XOR_KEY[i % XOR_KEY.length];
  }
  return result;
}

function parseHeader(view: DataView): HzModelHeader {
  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error('Invalid .hzmodel file: bad magic');
  }
  return {
    version: view.getUint16(4, true),
    flags: view.getUint16(6, true),
    vocabOffset: view.getUint32(8, true),
    vocabLength: view.getUint32(12, true),
    weightsOffset: view.getUint32(16, true),
    weightsLength: view.getUint32(20, true),
    numLayers: view.getUint32(24, true),
    hiddenSize: view.getUint32(28, true),
    numClasses: view.getUint32(32, true),
    numFeatures: view.getUint32(36, true),
  };
}

/** Parse a .hzmodel binary into header, vocab, and weight data. */
export function parseHzModel(buffer: ArrayBuffer): HzModel {
  const view = new DataView(buffer);
  const header = parseHeader(view);

  // Parse vocab
  const vocabBytes = new Uint8Array(buffer, header.vocabOffset, header.vocabLength);
  const vocabText = new TextDecoder().decode(vocabBytes);
  const vocab = vocabText.split('\n');

  // Deobfuscate weights
  const rawWeights = new Uint8Array(buffer, header.weightsOffset, header.weightsLength);
  const weights = xorDeobfuscate(rawWeights);

  return { header, vocab, weights };
}

/** Fetch and parse a .hzmodel file from a URL. */
export async function loadHzModel(url: string): Promise<HzModel> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch model: ${resp.status} ${resp.statusText}`);
  }
  const buffer = await resp.arrayBuffer();
  return parseHzModel(buffer);
}
