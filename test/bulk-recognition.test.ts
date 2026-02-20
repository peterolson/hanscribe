/**
 * Bulk recognition test using makemeahanzi median stroke data.
 *
 * Downloads graphics.txt from the makemeahanzi repo, extracts medians for
 * characters in our vocabulary, converts them to stroke format, and verifies
 * the model recognizes them.
 *
 * Run with: npx vitest run test/bulk-recognition.test.ts
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync, writeFileSync } from 'fs';
import { execSync } from 'child_process';
import { parseHzModel } from '../src/model-loader';
import { parseWeights, runInference } from '../src/inference';
import { preprocessStrokes } from '../src/preprocessing';

// Path to cached medians data
const MEDIANS_CACHE = 'test/medians-cache.json';
const GRAPHICS_URL = 'https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt';

interface MedianEntry {
  character: string;
  medians: number[][][]; // [stroke][point][x,y]
}

/**
 * Convert makemeahanzi medians to our stroke format: [x, y, t][].
 * makemeahanzi uses an SVG coordinate system where y increases upward,
 * but our model expects y increasing downward. Flip y.
 * Add synthetic timestamps spaced 8ms apart.
 */
function mediansToStrokes(medians: number[][][]): number[][][] {
  let t = 0;
  return medians.map(stroke => {
    return stroke.map(([x, y]) => {
      const point = [x, 900 - y, t]; // flip y, 900 is approx max
      t += 8;
      return point;
    });
  });
}

function loadMedians(): MedianEntry[] {
  if (existsSync(MEDIANS_CACHE)) {
    return JSON.parse(readFileSync(MEDIANS_CACHE, 'utf-8'));
  }

  // Download and parse graphics.txt
  console.log('Downloading makemeahanzi graphics.txt...');
  const raw = execSync(`curl -sL "${GRAPHICS_URL}"`, { maxBuffer: 50 * 1024 * 1024 }).toString();
  const entries: MedianEntry[] = [];
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    try {
      const obj = JSON.parse(line);
      if (obj.character && obj.medians) {
        entries.push({ character: obj.character, medians: obj.medians });
      }
    } catch { /* skip malformed lines */ }
  }

  writeFileSync(MEDIANS_CACHE, JSON.stringify(entries));
  console.log(`Cached ${entries.length} entries`);
  return entries;
}

describe('bulk recognition from makemeahanzi medians', () => {
  // Load model once
  const modelBytes = readFileSync('test/test.hzmodel');
  const buffer = modelBytes.buffer.slice(
    modelBytes.byteOffset,
    modelBytes.byteOffset + modelBytes.byteLength,
  );
  const model = parseHzModel(buffer);
  const weights = parseWeights(model);
  const vocabSet = new Set(weights.vocab);

  it('recognizes a sample of common characters at top-5', () => {
    const medians = loadMedians();

    // Filter to characters in our vocabulary
    const inVocab = medians.filter(e => vocabSet.has(e.character));
    console.log(`Characters in vocab: ${inVocab.length} / ${medians.length}`);

    // Test a random sample from the full vocab (deterministic seed)
    // Shuffle using a simple seeded PRNG for reproducibility
    const shuffled = [...inVocab];
    let seed = 42;
    for (let i = shuffled.length - 1; i > 0; i--) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      const j = seed % (i + 1);
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const SAMPLE_SIZE = 200;
    const toTest = shuffled.slice(0, SAMPLE_SIZE);
    console.log(`Testing ${toTest.length} characters...`);

    let top1Correct = 0;
    let top5Correct = 0;
    const failures: string[] = [];

    for (const entry of toTest) {
      const strokes = mediansToStrokes(entry.medians);
      // Skip if any stroke has fewer than 2 points
      if (strokes.some(s => s.length < 2)) continue;

      try {
        const { data, numSegments } = preprocessStrokes(strokes);
        if (numSegments === 0) continue;

        const { indices } = runInference(weights, data, numSegments, 5);
        const topChars = indices.map(i => weights.vocab[i]);

        if (topChars[0] === entry.character) {
          top1Correct++;
          top5Correct++;
        } else if (topChars.includes(entry.character)) {
          top5Correct++;
        } else {
          failures.push(`${entry.character} → ${topChars.join(',')}`);
        }
      } catch (err) {
        failures.push(`${entry.character} → ERROR: ${err}`);
      }
    }

    const tested = top1Correct + (top5Correct - top1Correct) + failures.length;
    console.log(`\nResults: ${tested} characters tested`);
    console.log(`  Top-1 accuracy: ${top1Correct}/${tested} (${(top1Correct/tested*100).toFixed(1)}%)`);
    console.log(`  Top-5 accuracy: ${top5Correct}/${tested} (${(top5Correct/tested*100).toFixed(1)}%)`);
    if (failures.length > 0) {
      console.log(`  Failures (${failures.length}):`);
      for (const f of failures) console.log(`    ${f}`);
    }

    // Expect at least 60% top-1 accuracy on clean median strokes
    expect(top1Correct / tested).toBeGreaterThan(0.6);
    // Expect at least 80% top-5 accuracy
    expect(top5Correct / tested).toBeGreaterThan(0.8);
  }, 300000); // 5-minute timeout for bulk test
});
