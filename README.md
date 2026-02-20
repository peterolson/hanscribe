# HanScribe

Chinese handwriting recognition for the web. Zero dependencies, small bundle (~7 KB gzipped), runs entirely in the browser.

## Features

- Draw Chinese characters on an HTML canvas and get back ranked character predictions
- 12,361 character vocabulary (GB2312 + common traditional characters)
- No server required — model runs client-side
- Automatic recognition after each stroke (debounced)
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

<div
    id="draw-area"
    style="width: 300px; height: 300px; border: 1px solid #ccc;"
></div>
<div id="results"></div>

<script>
    HanScribe.create({
        element: document.getElementById("draw-area"),
        onRecognize(results) {
            document.getElementById("results").textContent = results
                .map((r) => r.char + " " + (r.score * 100).toFixed(1) + "%")
                .join("  ");
        },
    });
</script>
```

### ES module (with bundler)

```js
import { HanScribe } from "hanscribe";

const pad = await HanScribe.create({
    element: document.getElementById("draw-area"),
    onRecognize(results) {
        document.getElementById("results").textContent = results
            .map((r) => `${r.char} ${(r.score * 100).toFixed(1)}%`)
            .join("  ");
    },
});
```

## Model File

The `.hzmodel` file (~7.5 MB) is not included in the npm package. By default, the library fetches it from [jsdelivr](https://cdn.jsdelivr.net/gh/peterolson/hanscribe@v0.1.1/test/test.hzmodel), so no configuration is needed for basic usage.

To self-host the model (recommended for production), download it from the [Releases page](https://github.com/peterolson/hanscribe/releases) and pass its URL via the `modelUrl` option:

```js
const pad = await HanScribe.create({
    element: document.getElementById("draw-area"),
    modelUrl: "/hanscribe.hzmodel",
});
```

## API Reference

### `HanScribe.create(options): Promise<HanScribeInstance>`

Creates a new handwriting recognition pad.

#### Options

| Option        | Type                                   | Default      | Description                                    |
| ------------- | -------------------------------------- | ------------ | ---------------------------------------------- |
| `element`     | `HTMLElement`                          | _required_   | Container element (canvas created inside)      |
| `onRecognize` | `(results: HanScribeResult[]) => void` | `undefined`  | Called with results after each stroke          |
| `modelUrl`    | `string`                               | jsdelivr CDN | URL to the .hzmodel file                       |
| `strokeColor` | `string`                               | `'#000'`     | Stroke color                                   |
| `strokeWidth` | `number`                               | `3`          | Stroke width in CSS pixels                     |
| `topK`        | `number`                               | `10`         | Number of top candidates to return             |
| `debounceMs`  | `number`                               | `300`        | Delay after stroke end before auto-recognizing |

#### HanScribeResult

```typescript
interface HanScribeResult {
    char: string; // The recognized character
    score: number; // Probability (0–1)
}
```

#### HanScribeInstance

```typescript
interface HanScribeInstance {
    clear(): void; // Clear strokes and canvas
    destroy(): void; // Clean up resources
    getStrokes(): number[][][]; // Get raw stroke data
    recognize(): Promise<HanScribeResult[]>; // Trigger recognition manually
}
```

## License

MIT
