/**
 * Web Worker that handles model loading, preprocessing, and inference.
 * All heavy computation happens here, keeping the main thread responsive.
 *
 * Inference runs in WASM (AssemblyScript) for ~4x speedup over TypeScript.
 */

import { parseHzModel } from "./model-loader";
import { createWasmEngine, type WasmInferenceEngine } from "./wasm-inference";
import { preprocessStrokes } from "./preprocessing";
import type { WorkerRequest, WorkerResponse, HanScribeResult } from "./types";

// @ts-ignore — virtual module resolved by rollup wasmInline plugin
import wasmBase64 from "wasm-inline";

let engine: WasmInferenceEngine | null = null;
let vocab: string[] = [];

/** Decode base64 string to Uint8Array. */
function decodeBase64(b64: string): Uint8Array {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i);
    }
    return bytes;
}

async function handleInit(modelUrl: string): Promise<void> {
    const resp = await fetch(modelUrl);
    if (!resp.ok) {
        throw new Error(
            `Failed to fetch model: ${resp.status} ${resp.statusText}`,
        );
    }
    const buffer = await resp.arrayBuffer();
    const model = parseHzModel(buffer);

    // Decode inlined WASM binary and create engine
    const wasmBytes = decodeBase64(wasmBase64);
    engine = await createWasmEngine(wasmBytes, model.header, model.weights);
    vocab = model.vocab;
}

function isCJK(codePoint: number): boolean {
    return (
        (0x3400 <= codePoint && codePoint <= 0x4dbf) || // CJK Unified Ideographs Extension A
        (0x4e00 <= codePoint && codePoint <= 0x9ffc) || // CJK Unified Ideographs
        (0xf900 <= codePoint && codePoint <= 0xfa6d) || // CJK Compatibility Ideographs
        (0xfa70 <= codePoint && codePoint <= 0xfad9) ||
        (0x20000 <= codePoint && codePoint <= 0x2a6dd) || // CJK Extension B
        (0x2a700 <= codePoint && codePoint <= 0x2b734) || // CJK Extension C
        (0x2b740 <= codePoint && codePoint <= 0x2b81d) || // CJK Extension D
        (0x2b820 <= codePoint && codePoint <= 0x2cea1) || // CJK Extension E
        (0x2ceb0 <= codePoint && codePoint <= 0x2ebe0) || // CJK Extension F
        (0x30000 <= codePoint && codePoint <= 0x3134a) || // CJK Extension G
        (0x2f800 <= codePoint && codePoint <= 0x2fa1d) // CJK Compatibility Supplement
    );
}

function handleRecognize(
    strokes: number[][][],
    topK: number,
): HanScribeResult[] {
    if (!engine) {
        throw new Error("Model not loaded");
    }

    const { data, numSegments } = preprocessStrokes(strokes);
    if (numSegments === 0) {
        return [];
    }

    // Request extra candidates so we still have topK after filtering non-CJK
    const { indices, scores } = engine.runInference(
        data,
        numSegments,
        topK * 2,
    );

    const results: HanScribeResult[] = [];
    for (let i = 0; i < indices.length && results.length < topK; i++) {
        const char = vocab[indices[i]];
        const cp = char.codePointAt(0);
        if (cp !== undefined && isCJK(cp)) {
            results.push({ char, score: scores[i] });
        }
    }
    return results;
}

function respond(msg: WorkerResponse) {
    (self as unknown as Worker).postMessage(msg);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
    const msg = e.data;

    try {
        switch (msg.type) {
            case "init":
                await handleInit(msg.modelUrl);
                respond({ type: "ready" });
                break;

            case "recognize": {
                const results = handleRecognize(msg.strokes, msg.topK);
                respond({ type: "result", results });
                break;
            }
        }
    } catch (err) {
        respond({ type: "error", message: String(err) });
    }
};
