/**
 * Canvas module — handles pointer events, stroke rendering, and high-DPI scaling.
 */

export interface CanvasOptions {
  strokeColor: string;
  strokeWidth: number;
  onStrokeEnd: () => void;
}

export interface CanvasController {
  clear(): void;
  getStrokes(): number[][][];
  destroy(): void;
}

export function createCanvas(
  container: HTMLElement,
  options: CanvasOptions,
): CanvasController {
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  canvas.style.touchAction = 'none'; // prevent scroll interference
  canvas.style.cursor = 'crosshair';
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d')!;
  const strokes: number[][][] = [];
  let currentStroke: number[][] | null = null;
  let dpr = 1;

  function resize() {
    dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.lineWidth = options.strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = options.strokeColor;

    // Redraw existing strokes after resize
    redraw();
  }

  function redraw() {
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.lineWidth = options.strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = options.strokeColor;

    for (const stroke of strokes) {
      if (stroke.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(stroke[0][0], stroke[0][1]);
      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i][0], stroke[i][1]);
      }
      ctx.stroke();
    }
  }

  function getPos(e: PointerEvent): [number, number, number] {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top, e.timeStamp];
  }

  function onPointerDown(e: PointerEvent) {
    canvas.setPointerCapture(e.pointerId);
    const [x, y, t] = getPos(e);
    currentStroke = [[x, y, t]];
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function onPointerMove(e: PointerEvent) {
    if (!currentStroke) return;
    const [x, y, t] = getPos(e);
    currentStroke.push([x, y, t]);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function onPointerEnd() {
    if (currentStroke && currentStroke.length > 1) {
      strokes.push(currentStroke);
      currentStroke = null;
      options.onStrokeEnd();
    } else {
      currentStroke = null;
    }
  }

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerEnd);
  canvas.addEventListener('pointercancel', onPointerEnd);

  const resizeObserver = new ResizeObserver(() => resize());
  resizeObserver.observe(container);

  // Initial sizing
  resize();

  return {
    clear() {
      strokes.length = 0;
      currentStroke = null;
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
    },
    getStrokes() {
      return strokes.map(s => s.map(p => [...p]));
    },
    destroy() {
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerEnd);
      canvas.removeEventListener('pointercancel', onPointerEnd);
      resizeObserver.disconnect();
      container.removeChild(canvas);
    },
  };
}
