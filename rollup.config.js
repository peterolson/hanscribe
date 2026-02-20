import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import terser from '@rollup/plugin-terser';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Rollup plugin that inlines the compiled WASM binary as a base64 string.
 * Resolves `import wasmBase64 from 'wasm-inline'` to the build output.
 */
function wasmInline() {
  return {
    name: 'wasm-inline',
    resolveId(id) {
      if (id === 'wasm-inline') return id;
      return null;
    },
    load(id) {
      if (id === 'wasm-inline') {
        const wasmPath = join(__dirname, 'build', 'inference.wasm');
        const binary = readFileSync(wasmPath);
        const b64 = binary.toString('base64');
        return `export default "${b64}";`;
      }
      return null;
    },
  };
}

/**
 * Rollup plugin that inlines the pre-built worker bundle as a string constant.
 * Resolves `import workerCode from 'worker-inline'` to the built worker IIFE.
 */
function workerInline() {
  return {
    name: 'worker-inline',
    resolveId(id) {
      if (id === 'worker-inline') return id;
      return null;
    },
    load(id) {
      if (id === 'worker-inline') {
        const workerPath = join(__dirname, 'build', 'worker.bundle.js');
        const code = readFileSync(workerPath, 'utf-8');
        // Escape backticks and backslashes for template literal
        const escaped = code.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$/g, '\\$');
        return `export default \`${escaped}\`;`;
      }
      return null;
    },
  };
}

export default [
  // Step 1: Build worker as self-contained IIFE (includes inlined WASM)
  {
    input: 'src/worker.ts',
    output: {
      file: 'build/worker.bundle.js',
      format: 'iife',
      sourcemap: false,
    },
    plugins: [
      wasmInline(),
      resolve(),
      typescript({ tsconfig: './tsconfig.json', declaration: false }),
      terser(),
    ],
  },
  // Step 2: ESM bundle (with inlined worker)
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/hanscribe.js',
      format: 'es',
      sourcemap: true,
    },
    plugins: [
      workerInline(),
      resolve(),
      typescript({
        tsconfig: './tsconfig.json',
        declaration: true,
        declarationDir: 'dist',
      }),
      terser(),
    ],
  },
  // Step 3: UMD bundle (with inlined worker)
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/hanscribe.umd.js',
      format: 'umd',
      name: 'HanScribe',
      exports: 'named',
      outro: 'Object.assign(exports, exports.HanScribe);',
      sourcemap: true,
    },
    plugins: [
      workerInline(),
      resolve(),
      typescript({
        tsconfig: './tsconfig.json',
      }),
      terser(),
    ],
  },
];
