import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

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
  // Step 1: Build worker as self-contained IIFE
  {
    input: 'src/worker.ts',
    output: {
      file: 'build/worker.bundle.js',
      format: 'iife',
      sourcemap: false,
    },
    plugins: [
      resolve(),
      typescript({ tsconfig: './tsconfig.json', declaration: false }),
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
      sourcemap: true,
    },
    plugins: [
      workerInline(),
      resolve(),
      typescript({
        tsconfig: './tsconfig.json',
      }),
    ],
  },
];
