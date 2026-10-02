/** scene visualizer — a reconstruction in one orbiting three.js view:
 *
 *   - sparse points  (P, 3), optional colors (P, 3) in 0..1
 *   - cameras        (F, 3, 4) world → camera [R | t]: a frustum per camera
 *                    (first one highlighted) + the path through the centres;
 *                    frustum shape from intrinsics (3, 3, pixels) + image_size
 *                    [W, H] when given, else a 60° 4:3 default
 *   - dense points   (N, 3) + colors (N, 3) + dense_frames (N,) — e.g. depth
 *                    maps back-projected per frame; colorable by frame to see
 *                    which frame each layer came from
 *
 *  Coordinates are OpenCV camera conventions (x right, y down, z forward);
 *  the view flips y and z so "up" is up and camera 0 looks into the screen.
 *  Nothing here knows a box: the def's params name the keys.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SerValue } from "../api";
import { fetchTyped, flattenNumbers, isRef } from "../resolvers";
import { heatColor } from "./MatrixHeatmap";

type Num = ArrayLike<number>;

interface SceneData {
  points: Float32Array | null;
  colors: Float32Array | null;
  cameras: Num | null;            // F × 12, row-major [R | t]
  dense: Float32Array | null;
  denseColors: Float32Array | null;
  denseFrames: Num | null;
  K: Num | null;
  size: [number, number] | null;
}

async function numbers(v: unknown): Promise<Num | null> {
  if (v === undefined || v === null) return null;
  if (isRef(v)) {
    const t = await fetchTyped(v as SerValue);
    return t && t.data.length > 0 ? t.data : null;
  }
  return flattenNumbers(v);
}

function f32(a: Num | null): Float32Array | null {
  if (!a) return null;
  return a instanceof Float32Array ? a : Float32Array.from(a);
}

/** OpenCV (x, y, z) → view (x, −y, −z), in place. */
function toView(a: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i += 3) { a[i + 1] = -a[i + 1]; a[i + 2] = -a[i + 2]; }
  return a;
}

function rgbOf(css: string): [number, number, number] {
  const m = css.match(/\d+/g);
  return m ? [Number(m[0]) / 255, Number(m[1]) / 255, Number(m[2]) / 255] : [1, 1, 1];
}

export function SceneView({
  title, points, colors, cameras, densePoints, denseColors, denseFrames, intrinsics, imageSize,
}: {
  title?: string;
  points: unknown;
  colors?: unknown;
  cameras?: unknown;
  densePoints?: unknown;
  denseColors?: unknown;
  denseFrames?: unknown;
  intrinsics?: unknown;
  imageSize?: unknown;
}) {
  const [data, setData] = useState<SceneData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showSparse, setShowSparse] = useState(true);
  const [showDense, setShowDense] = useState(true);
  const [showCams, setShowCams] = useState(true);
  const [byFrame, setByFrame] = useState(false);
  const [ptSize, setPtSize] = useState(1);

  useEffect(() => {
    let alive = true;
    setData(null);
    setErr(null);
    void (async () => {
      try {
        const [p, c, cam, d, dc, df, K] = await Promise.all([
          numbers(points), numbers(colors), numbers(cameras),
          numbers(densePoints), numbers(denseColors), numbers(denseFrames), numbers(intrinsics),
        ]);
        const sz = flattenNumbers(imageSize);
        if (!alive) return;
        if (!p && !d) { setErr("no points in this response"); return; }
        setData({
          points: p ? toView(Float32Array.from(p)) : null,
          colors: f32(c),
          cameras: cam,
          dense: d ? toView(Float32Array.from(d)) : null,
          denseColors: f32(dc),
          denseFrames: df,
          K: K && K.length >= 9 ? K : null,
          size: sz && sz.length >= 2 ? [sz[0], sz[1]] : null,
        });
      } catch (e) {
        if (alive) setErr(String((e as Error).message || e));
      }
    })();
    return () => { alive = false; };
  }, [points, colors, cameras, densePoints, denseColors, denseFrames, intrinsics, imageSize]);

  const nCams = data?.cameras ? Math.floor(data.cameras.length / 12) : 0;
  const nSparse = data?.points ? data.points.length / 3 : 0;
  const nDense = data?.dense ? data.dense.length / 3 : 0;

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      {err && <div className="note">{err}</div>}
      {!data && !err && <span className="spinnerbox"><span className="spinner" /></span>}
      {data && (
        <>
          <div className="btnrow" style={{ marginBottom: 6, flexWrap: "wrap", gap: 10, fontSize: 12.5 }}>
            {nSparse > 0 && (
              <label><input type="checkbox" checked={showSparse} onChange={(e) => setShowSparse(e.target.checked)} /> tracks ({nSparse.toLocaleString()})</label>
            )}
            {nDense > 0 && (
              <label><input type="checkbox" checked={showDense} onChange={(e) => setShowDense(e.target.checked)} /> depth ({nDense.toLocaleString()})</label>
            )}
            {nCams > 0 && (
              <label><input type="checkbox" checked={showCams} onChange={(e) => setShowCams(e.target.checked)} /> cameras ({nCams})</label>
            )}
            {nDense > 0 && data.denseFrames && (
              <label title="color the depth points by the frame they came from (viridis: first → last)">
                <input type="checkbox" checked={byFrame} onChange={(e) => setByFrame(e.target.checked)} /> color depth by frame
              </label>
            )}
            <label>point size
              <input type="range" min={0.25} max={4} step={0.25} value={ptSize}
                onChange={(e) => setPtSize(Number(e.target.value))} style={{ width: 90, marginLeft: 6 }} />
            </label>
          </div>
          <SceneCanvas data={data} showSparse={showSparse} showDense={showDense}
            showCams={showCams} byFrame={byFrame} ptSize={ptSize} />
          <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>
            drag: orbit · right-drag: pan · wheel: zoom · first camera = origin (highlighted)
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

/** Scene extent: 2–98 % range of the sparse points (dense when there are
 *  none) — robust to a few far-off points — grown to hold every camera
 *  centre (cameras may orbit the points). */
function extent(pts: Float32Array, centres: number[][]): { center: THREE.Vector3; size: number } {
  const lo: number[] = [], hi: number[] = [];
  const n = pts.length / 3;
  const step = Math.max(1, Math.floor(n / 20000));
  for (let c = 0; c < 3; c++) {
    const vals: number[] = [];
    for (let i = 0; i < n; i += step) { const x = pts[i * 3 + c]; if (Number.isFinite(x)) vals.push(x); }
    vals.sort((a, b) => a - b);
    let l = vals.length ? vals[Math.floor(0.02 * (vals.length - 1))] : Infinity;
    let h = vals.length ? vals[Math.ceil(0.98 * (vals.length - 1))] : -Infinity;
    for (const C of centres) { l = Math.min(l, C[c]); h = Math.max(h, C[c]); }
    lo.push(Number.isFinite(l) ? l : -1);
    hi.push(Number.isFinite(h) ? h : 1);
  }
  return {
    center: new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2),
    size: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-6),
  };
}

function SceneCanvas({
  data, showSparse, showDense, showCams, byFrame, ptSize,
}: {
  data: SceneData; showSparse: boolean; showDense: boolean; showCams: boolean;
  byFrame: boolean; ptSize: number;
}) {
  const holderRef = useRef<HTMLDivElement | null>(null);
  const objs = useRef<{ sparse?: THREE.Points; dense?: THREE.Points; cams?: THREE.Group;
    denseColor?: THREE.BufferAttribute; frameColor?: THREE.BufferAttribute;
    base?: { sparse: number; dense: number } }>({});
  const [failed, setFailed] = useState<string | null>(null);

  // camera centres + frustum geometry (view coordinates)
  const camGeo = useMemo(() => {
    const cams = data.cameras;
    if (!cams) return null;
    const F = Math.floor(cams.length / 12);
    const centres: number[][] = [];
    const Rs: number[][] = [], ts: number[][] = [];
    for (let f = 0; f < F; f++) {
      const m = Array.from({ length: 12 }, (_, i) => Number(cams[f * 12 + i]));
      const R = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
      const t = [m[3], m[7], m[11]];
      // C = −Rᵀ t
      centres.push([0, 1, 2].map((j) => -(R[j] * t[0] + R[3 + j] * t[1] + R[6 + j] * t[2])));
      Rs.push(R); ts.push(t);
    }
    return { F, centres, Rs, ts };
  }, [data.cameras]);

  useEffect(() => {
    const holder = holderRef.current;
    if (!holder) return;
    let disposed = false, raf = 0;
    const disposables: { dispose: () => void }[] = [];
    let renderer: THREE.WebGLRenderer, controls: OrbitControls;
    try {
      const centresView = (camGeo?.centres ?? []).map((c) => [c[0], -c[1], -c[2]]);
      const ext = extent(data.points ?? data.dense ?? new Float32Array(0), centresView);
      const size = ext.size;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#0a0d11");
      const camera = new THREE.PerspectiveCamera(50, holder.clientWidth / holder.clientHeight, size / 1e4, size * 1e3);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(holder.clientWidth, holder.clientHeight);
      holder.appendChild(renderer.domElement);

      const root = new THREE.Group();
      root.position.copy(ext.center.clone().negate());
      scene.add(root);

      const mkPoints = (pos: Float32Array, col: Float32Array | null, n: number, sz: number, fallback: string) => {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        let colorAttr: THREE.BufferAttribute | undefined;
        if (col && col.length >= n * 3) {
          colorAttr = new THREE.BufferAttribute(col, 3);
          g.setAttribute("color", colorAttr);
        }
        const m = new THREE.PointsMaterial({
          size: sz, sizeAttenuation: true, vertexColors: !!colorAttr,
          color: colorAttr ? 0xffffff : new THREE.Color(fallback),
        });
        disposables.push(g, m);
        return { pts: new THREE.Points(g, m), colorAttr };
      };

      const base = { sparse: size / 120, dense: size / 400 };
      objs.current.base = base;
      if (data.dense) {
        const n = data.dense.length / 3;
        const { pts, colorAttr } = mkPoints(data.dense, data.denseColors, n, base.dense, "#8aa4c8");
        objs.current.dense = pts;
        objs.current.denseColor = colorAttr;
        if (data.denseFrames && data.denseFrames.length === n) {
          let maxF = 0;
          for (let i = 0; i < n; i++) maxF = Math.max(maxF, Number(data.denseFrames[i]));
          const F = Math.max(1, (camGeo?.F ?? 0) || maxF + 1);
          const lut = Array.from({ length: F }, (_, f) => rgbOf(heatColor(F > 1 ? f / (F - 1) : 0.5)));
          const fc = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) {
            const c = lut[Math.min(F - 1, Math.max(0, Math.round(Number(data.denseFrames[i]))))];
            fc[i * 3] = c[0]; fc[i * 3 + 1] = c[1]; fc[i * 3 + 2] = c[2];
          }
          objs.current.frameColor = new THREE.BufferAttribute(fc, 3);
        }
        root.add(pts);
      }
      if (data.points) {
        const n = data.points.length / 3;
        const { pts } = mkPoints(data.points, data.colors, n, base.sparse, "#ffd166");
        objs.current.sparse = pts;
        root.add(pts);
      }

      if (camGeo && camGeo.F > 0) {
        const group = new THREE.Group();
        const [W, H] = data.size ?? [4, 3];
        const K = data.K ? Array.from(data.K).map(Number) : [W / (2 * Math.tan(Math.PI / 6)), 0, W / 2, 0, W / (2 * Math.tan(Math.PI / 6)), H / 2, 0, 0, 1];
        const L = size * 0.06;
        const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([u, v]) => [(u - K[2]) / K[0] * L, (v - K[5]) / K[4] * L, L]);
        const segs: number[] = [], segs0: number[] = [], path: number[] = [];
        for (let f = 0; f < camGeo.F; f++) {
          const R = camGeo.Rs[f], t = camGeo.ts[f], C = camGeo.centres[f];
          // world point of a camera-frame point x: Rᵀ (x − t)
          const w = corners.map((x) => [0, 1, 2].map((j) =>
            R[j] * (x[0] - t[0]) + R[3 + j] * (x[1] - t[1]) + R[6 + j] * (x[2] - t[2])));
          const out = f === 0 ? segs0 : segs;
          const push = (a: number[], b: number[]) => out.push(a[0], -a[1], -a[2], b[0], -b[1], -b[2]);
          for (let i = 0; i < 4; i++) { push(C, w[i]); push(w[i], w[(i + 1) % 4]); }
          path.push(C[0], -C[1], -C[2]);
        }
        const line = (arr: number[], color: string, strip = false) => {
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(arr), 3));
          const m = new THREE.LineBasicMaterial({ color });
          disposables.push(g, m);
          return strip ? new THREE.Line(g, m) : new THREE.LineSegments(g, m);
        };
        if (segs.length) group.add(line(segs, "#5ba8ff"));
        group.add(line(segs0, "#ff6b6b"));
        if (path.length > 3) group.add(line(path, "#e8edf3", true));
        objs.current.cams = group;
        root.add(group);
      }

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      // start behind and slightly above the first camera, looking at the scene
      camera.position.set(size * 0.15, size * 0.35, size * 1.3);
      controls.target.set(0, 0, 0);
      controls.update();

      const loop = () => {
        if (disposed) return;
        controls.update();
        renderer.render(scene, camera);
        raf = requestAnimationFrame(loop);
      };
      loop();

      const ro = new ResizeObserver(() => {
        if (disposed || !holder.clientWidth) return;
        camera.aspect = holder.clientWidth / holder.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(holder.clientWidth, holder.clientHeight);
      });
      ro.observe(holder);

      return () => {
        disposed = true;
        cancelAnimationFrame(raf);
        ro.disconnect();
        controls.dispose();
        renderer.dispose();
        for (const d of disposables) d.dispose();
        if (renderer.domElement.parentNode === holder) holder.removeChild(renderer.domElement);
        objs.current = {};
      };
    } catch (e) {
      setFailed(String((e as Error).message || e));
      return undefined;
    }
  }, [data, camGeo]);

  // toggles: no scene rebuild
  useEffect(() => {
    const o = objs.current;
    if (o.sparse) {
      o.sparse.visible = showSparse;
      (o.sparse.material as THREE.PointsMaterial).size = (o.base?.sparse ?? 1) * ptSize;
    }
    if (o.dense) {
      o.dense.visible = showDense;
      const m = o.dense.material as THREE.PointsMaterial;
      m.size = (o.base?.dense ?? 1) * ptSize;
      const want = byFrame && o.frameColor ? o.frameColor : o.denseColor;
      if (want && o.dense.geometry.getAttribute("color") !== want) {
        o.dense.geometry.setAttribute("color", want);
        m.vertexColors = true;
        m.color.set(0xffffff);
        m.needsUpdate = true;
      }
    }
    if (o.cams) o.cams.visible = showCams;
  }, [showSparse, showDense, showCams, byFrame, ptSize, data, camGeo]);

  return (
    <div>
      <div className="glbview" style={{ height: 520 }} ref={holderRef} />
      {failed && <div className="note" style={{ color: "var(--err)" }}>scene render failed: {failed}</div>}
    </div>
  );
}
