# HanScribe

Chinese handwriting recognition for the web. Zero dependencies, small bundle (~7 KB gzipped), runs entirely in the browser.

## Features

- Draw Chinese characters on an HTML canvas and get back ranked character predictions
- 12,361 character vocabulary (GB2312 + common traditional characters)
- No server required — model runs client-side
- Automatic recognition after each stroke (debounced)
- High-DPI canvas support
- TypeScript types included

## Installation

```bash
npm install hanscribe
```

Or include via script tag — see [Quick Start](#quick-start) below.

## Quick Start

### Script tag (no bundler)

```html
<script src="https://unpkg.com/hanscribe/dist/hanscribe.umd.js"></script>

<div id="draw-area" style="width: 300px; height: 300px; border: 1px solid #ccc;"></div>
<div id="results"></div>

<script>
HanScribe.HanScribe.create({
  element: document.getElementById('draw-area'),
  onRecognize(results) {
    document.getElementById('results').textContent =
      results.map(r => r.char + ' ' + (r.score * 100).toFixed(1) + '%').join('  ');
  },
});
</script>
```

### ES module (with bundler)

```js
import { HanScribe } from 'hanscribe';

const pad = await HanScribe.create({
  element: document.getElementById('draw-area'),
  onRecognize(results) {
    document.getElementById('results').textContent =
      results.map(r => `${r.char} ${(r.score * 100).toFixed(1)}%`).join('  ');
  },
});
```

## Model File

The `.hzmodel` file (~7.5 MB) is not included in the npm package. By default, the library fetches it from [jsdelivr](https://cdn.jsdelivr.net/gh/peterolson/hanscribe@master/test/test.hzmodel), so no configuration is needed for basic usage.

To self-host the model (recommended for production), download it from the [Releases page](https://github.com/peterolson/hanscribe/releases) and pass its URL via the `modelUrl` option:

```js
const pad = await HanScribe.create({
  element: document.getElementById('draw-area'),
  modelUrl: '/hanscribe.hzmodel',
});
```

## API Reference

### `HanScribe.create(options): Promise<HanScribeInstance>`

Creates a new handwriting recognition pad.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `element` | `HTMLElement` | *required* | Container element (canvas created inside) |
| `onRecognize` | `(results: HanScribeResult[]) => void` | `undefined` | Called with results after each stroke |
| `modelUrl` | `string` | jsdelivr CDN | URL to the .hzmodel file |
| `strokeColor` | `string` | `'#000'` | Stroke color |
| `strokeWidth` | `number` | `3` | Stroke width in CSS pixels |
| `topK` | `number` | `10` | Number of top candidates to return |
| `debounceMs` | `number` | `300` | Delay after stroke end before auto-recognizing |

#### HanScribeResult

```typescript
interface HanScribeResult {
  char: string;   // The recognized character
  score: number;  // Probability (0–1)
}
```

#### HanScribeInstance

```typescript
interface HanScribeInstance {
  clear(): void;                              // Clear strokes and canvas
  destroy(): void;                            // Clean up resources
  getStrokes(): number[][][];                 // Get raw stroke data
  recognize(): Promise<HanScribeResult[]>;    // Trigger recognition manually
}
```

### Advanced: Using Individual Modules

```typescript
import { preprocessStrokes, parseHzModel, parseWeights, runInference } from 'hanscribe';

// Load model
const resp = await fetch('/hanscribe.hzmodel');
const model = parseHzModel(await resp.arrayBuffer());
const weights = parseWeights(model);

// Process strokes (array of strokes, each stroke is array of [x, y, timestamp])
const strokes = [[[100, 50, 0], [100, 200, 100]]];
const { data, numSegments } = preprocessStrokes(strokes);

// Run inference
const { indices, scores } = runInference(weights, data, numSegments, 10);
const topChar = weights.vocab[indices[0]];
console.log(topChar, scores[0]);
```

## Browser Support

Requires: Canvas 2D, PointerEvents, ResizeObserver, Float32Array. Works in all modern browsers (Chrome 64+, Firefox 59+, Safari 13+, Edge 79+).

## How It Works

1. **Stroke capture**: PointerEvents on an HTML canvas
2. **Bezier fitting**: Schneider's algorithm fits cubic Bezier curves to raw points
3. **Feature extraction**: 10-dimensional features per Bezier segment (direction, curvature, pen state)
4. **Neural network**: 4-layer bidirectional LSTM (192 units) + fully-connected classifier
5. **Decoding**: Softmax + max-probability aggregation across timesteps

## License

MIT
