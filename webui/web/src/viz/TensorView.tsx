/** Tensor/array view — the end-user "tensor visualizer":
 *
 *    * dtype / shape / size
 *    * min / max over the finite values
 *    * a value-distribution histogram (shape-agnostic, safe for any tensor)
 *    * a **spatial preview** when the box def declares one via
 *      ``params.spatial``:
 *        - ``"flow"`` -> quiver of the dense (u,v) field over the base image
 *          (def ``base`` names the input field) — (N,2,H,W) / (2,H,W) / (H,W,2)
 *        - ``"heat"`` -> viridis heat image of an (H, W) field
 *    * a one-click .npy download with the array's *true* dtype
 *
 *  Design rule (SPA-wide): the visualizer never *guesses* that a shape is
 *  spatial — (n_samples, hidden) embeddings, (frames, points, 2) tracks and
 *  real image maps are indistinguishable from shape alone, so the semantics
 *  are declared by the box definition (``params.spatial``), not inferred.
 *  Without the declaration every tensor still gets stats + histogram + .npy.
 *
 *  Bandwidth rule: inline values (``kind: "array"``) already carry their
 *  numbers in the API JSON, so everything is free.  Buffer artifacts
 *  (``kind: "buffer"``) are fetched *once*, only when they fit under the
 *  64 MB preview cap (same cap ``statsOf`` uses); larger ones stay
 *  metadata-only and the .npy download remains the explicit path.
 *
 *  The .npy is written with the array's **true dtype** (not assumed float32):
 *  a float32 tensor → ``<f4``; a bool tensor (e.g. TAPNext ``visibles``) →
 *  ``|b1`` (1-byte, 0/1); ints → their signed size.  This matters because a
 *  bool/integer tensor mis-labelled as float would both load with the wrong
 *  dtype and, for a bool inline array, lose every value. */
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { fetchTyped, isRef, shapeStr, statsOf, inlineValues } from "../resolvers";
import { bytesShort } from "../ui";
import { FlowField } from "./FlowField";

// ------------------------------------------------------------------ helpers

const PREVIEW_CAP_BYTES = 64 * 1024 * 1024;   // same cap as statsOf

/** Collect numeric *and* boolean leaves into a flat number array in
 *  row-major order (bool → 0/1).  This is what lets a bool inline tensor
 *  (JSON booleans) round-trip into a proper 1-byte bool .npy. */
function collect(v: unknown, out: number[] = []): number[] {
  if (typeof v === "number" && Number.isFinite(v)) { out.push(v); return out; }
  if (typeof v === "boolean") { out.push(v ? 1 : 0); return out; }
  if (Array.isArray(v)) { for (const x of v) collect(x, out); return out; }
  return out;
}

/** dtype → numpy descr + itemsize (bytes per element).  These are the two
 *  facts we need to write a correct .npy header.  The concrete TypedArray
 *  is chosen by ``encodeFlat`` below when re-encoding an inline value. */
const DTYPE: Record<string, { descr: string; itemsize: number }> = {
  bool:      { descr: "|b1", itemsize: 1 },
  "int8":    { descr: "<i1", itemsize: 1 },
  "int16":   { descr: "<i2", itemsize: 2 },
  "int32":   { descr: "<i4", itemsize: 4 },
  "int64":   { descr: "<i8", itemsize: 8 },   // best-effort: encoded via float64
  "uint8":   { descr: "|u1", itemsize: 1 },
  "uint16":  { descr: "<u2", itemsize: 2 },
  "uint32":  { descr: "<u4", itemsize: 4 },
  "float16": { descr: "<f2", itemsize: 2 },   // best-effort: encoded via float32
  "float32": { descr: "<f4", itemsize: 4 },
  "float64": { descr: "<f8", itemsize: 8 },
};
function dtypeInfo(name: string) { return DTYPE[name] ?? DTYPE.float32; }

function flatShape(shape: number[]): number {
  let p = 1; for (const s of shape) p *= s; return p;
}

/** Re-encode a flat number[] (bool → 0/1 already folded by ``collect``)
 *  into the little-endian byte layout for the declared dtype.
 *
 *  int64 and float16 are not representable in a single JS numeric, so we
 *  store their values in the closest larger native array (int64 → float64,
 *  float16 → float32).  The header still declares the *original* dtype, and
 *  a size mismatch is logged so the consumer knows about it.  For the dtypes
 *  our boxes actually use (bool / float32) this is exact. */
function encodeFlat(flat: number[], dtypeName: string): Uint8Array {
  let ta: Uint8Array | Uint16Array | Uint32Array
      | Int8Array  | Int16Array | Int32Array
      | Float32Array | Float64Array;
  switch (dtypeName) {
    case "bool":
    case "uint8":   ta = new Uint8Array(flat);   break;
    case "int8":    ta = new Int8Array(flat);    break;
    case "uint16":  ta = new Uint16Array(flat);  break;
    case "int16":   ta = new Int16Array(flat);   break;
    case "uint32":  ta = new Uint32Array(flat);  break;
    case "int32":   ta = new Int32Array(flat);   break;
    case "int64":   ta = new Float64Array(flat); break;   // best-effort
    case "float64": ta = new Float64Array(flat); break;
    case "float16": ta = new Float32Array(flat); break;   // best-effort
    default:        ta = new Float32Array(flat); break;   // float32 / unknown
  }
  const expected = flat.length * dtypeInfo(dtypeName).itemsize;
  if (ta.byteLength !== expected) {
    console.warn(`[tensor] inline encode ${ta.byteLength} B != declared ${expected} B (dtype=${dtypeName})`);
  }
  return new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
}

/**
 * Write a little-endian .npy (v1.0, C-order) for an array of the given dtype
 * and trigger a browser download.  ``payload`` is the raw element bytes in
 *  that dtype (a Uint8Array of 0/1 for bool, a Float32Array payload for
 *  float32, …).  For buffer artifacts we pass the fetched payload verbatim so
 *  no re-encode is needed.
 */
function saveNpy(payload: Uint8Array, dtypeName: string, shape: number[], filename: string) {
  const info = dtypeInfo(dtypeName);
  const n = payload.length;

  const tuple = shape.length === 0 ? "()" :
    "(" + shape.map(String).join(", ") + (shape.length === 1 ? "," : "") + ")";
  const base = `{'descr': '${info.descr}', 'fortran_order': False, 'shape': ${tuple}, }`;

  const PREFIX = 10;                       // 6 magic + 2 version + 2 header_len
  const minK   = Math.ceil((base.length + PREFIX) / 64);
  const target = 64 * minK - PREFIX;       // total header_len
  const full   = base + " ".repeat(Math.max(0, target - base.length - 1)) + "\n";
  const hb     = new TextEncoder().encode(full);
  // invariant: 10 + hb.length = 64*minK (a multiple of 64) — numpy requirement

  const out = new Uint8Array(PREFIX + hb.length + n);
  const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];   // \x93NUMPY
  for (let i = 0; i < 6; i++) out[i] = MAGIC[i];
  out[6] = 1; out[7] = 0;
  new DataView(out.buffer).setUint16(8, hb.length, true);
  out.set(hb, PREFIX);
  out.set(payload, PREFIX + hb.length);

  const blob = new Blob([out.buffer], { type: "application/octet-stream" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// --------------------------------------------------------- value histogram

interface Dist {
  mn: number; mx: number;
  invalid: number;                   // rough count of non-finite cells
  bins: number[] | null;   // NB values in [0,1], low → high values; null = all invalid
}

function summarize(values: ArrayLike<number>, NB = 36): Dist {
  const n = values.length;
  const step = Math.max(1, Math.floor(n / 50000));    // cap the scan
  let mn = Infinity, mx = -Infinity, invalid = 0, scanned = 0;
  for (let i = 0; i < n; i += step) {
    const v = Number(values[i]);
    if (!Number.isFinite(v)) { invalid += step; continue; }
    scanned++;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (scanned === 0) return { mn: NaN, mx: NaN, invalid: n, bins: null };
  const span = (mx - mn) || 1;
  const counts = new Array(NB).fill(0) as number[];
  for (let i = 0; i < n; i += step) {
    const v = Number(values[i]);
    if (!Number.isFinite(v)) continue;
    counts[Math.min(NB - 1, Math.floor((v - mn) / span * NB))]++;
  }
  const peak = Math.max(1, ...counts);
  return { mn, mx, invalid, bins: counts.map((c) => c / peak) };
}

function DistPlot({ d, label }: { d: Dist; label: string }) {
  if (!d.bins) {
    return <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11, marginTop: 10 }}>
      {label}: no finite values
    </div>;
  }
  const NB = d.bins.length;
  const W = 280, H = 48, pad = 3;
  const bw = (W - pad * 2) / NB;
  return (
    <div style={{ marginTop: 10 }}>
      <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-dim)" }}>
        {label} · min {d.mn.toPrecision(3)} … max {d.mx.toPrecision(3)}
        {d.invalid > 0 && ` · ${d.invalid} non-finite`}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: 320, display: "block", marginTop: 4 }}
        role="img" aria-label={label}
      >
        {d.bins.map((b, i) => (
          <rect key={i}
            x={pad + i * bw + 0.5}
            y={H - 8 - (H - 16) * b}
            width={Math.max(0.6, bw - 1)}
            height={Math.max(0.6, (H - 16) * b)}
            fill={`hsl(${195 + 115 * i / NB}, 65%, ${32 + 40 * b}%)`} />
        ))}
        <line x1={pad} y1={H - 8} x2={W - pad} y2={H - 8} stroke="var(--border)" />
      </svg>
    </div>
  );
}

// ------------------------------------------------------------- heat preview

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];
function heatRgb(t: number): [number, number, number] {
  if (!Number.isFinite(t)) return VIRIDIS[0];
  const c = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(c));
  const f = c - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
export const HEAT_GRADIENT =
  `linear-gradient(to right, ${VIRIDIS.map(([r, g, b]) => `rgb(${r},${g},${b})`).join(", ")})`;

function HeatPreview({ values, H, W }: { values: ArrayLike<number>; H: number; W: number }) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const ctx = off.getContext("2d");
    if (!ctx) return;
    const im = ctx.createImageData(W, H);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < H * W; i++) {
      const v = Number(values[i]);
      if (Number.isFinite(v)) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (!Number.isFinite(mn)) { mn = 0; mx = 1; }
    const span = (mx - mn) || 1;
    for (let i = 0; i < H * W; i++) {
      const v = Number(values[i]);
      const [r, g, b] = Number.isFinite(v) ? heatRgb((v - mn) / span) : [16, 21, 28];
      im.data[i * 4 + 0] = r; im.data[i * 4 + 1] = g; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    // draw downscaled into a data URL so the canvas scales responsively
    const scale = Math.min(1, 768 / Math.max(H, W));
    const disp = document.createElement("canvas");
    disp.width = Math.max(2, Math.round(W * scale));
    disp.height = Math.max(2, Math.round(H * scale));
    disp.getContext("2d")?.drawImage(off, 0, 0, disp.width, disp.height);
    setUrl(disp.toDataURL("image/png"));
  }, [values, H, W]);

  if (!url) return <span className="spinnerbox"><span className="spinner" /></span>;
  return <img src={url} alt="heat preview" style={{ maxWidth: "100%", display: "block" }} />;
}

// ------------------------------------------------------------------ component

export function TensorView({ value, title, base, spatial }: {
  value: unknown;
  title?: string;
  /** base image URL (e.g. first uploaded input) shown under the "flow" preview */
  base?: string | null;
  /** def-declared spatial semantics: "flow" (dense (u,v)) or "heat" ((H,W)) */
  spatial?: string | null;
}) {
  const [head, setHead]         = useState<number[] | null>(null);
  const [kind, setKind]         = useState<string | null>(null);
  const [dtype, setDtype]       = useState<string>("");
  const [shape, setShape]       = useState<string>("scalar");
  const [size, setSize]         = useState<number | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [dist, setDist]         = useState<Dist | null>(null);
  const [flat, setFlat]         = useState<Float32Array | null>(null);
  const [busy, setBusy]         = useState(true);
  const [capped, setCapped]     = useState(false);

  const isBuffer = isRef(value) && (value as { kind?: string }).kind === "buffer";
  const isInline = isRef(value) && (value as { kind?: string }).kind === "array";
  const rawRef   = isRef(value) ? (value as { dtype?: string; shape?: number[]; size?: number }) : null;
  const dtypeName = rawRef?.dtype || "float32";          // declared by the serializer (bool / float32 / …)
  const shapeArr  = rawRef?.shape ?? [];
  const tupleKey  = shapeArr.length ? shapeArr.map(String).join("x") : "scalar";
  const npyFileName = `tensor_${tupleKey}_${dtypeName}.npy`;
  const url = isRef(value) ? (value as { url?: string }).url : null;
  const sizeBytes = typeof rawRef?.size === "number" ? rawRef.size : null;

  // metadata (always available from the serializer for refs)
  useEffect(() => {
    if (isRef(value)) {
      const r = value as { kind?: string; dtype?: string; shape?: number[]; size?: number };
      setKind(String(r.kind || ""));
      setDtype(String(r.dtype || ""));
      setShape(shapeStr(r.shape));
      setSize(typeof r.size === "number" ? r.size : null);
    }
  }, [value]);

  // numbers (stats + previews) — one bounded fetch for buffers
  useEffect(() => {
    let alive = true;
    setHead(null); setDist(null); setFlat(null); setBusy(true); setCapped(false);
    (async () => {
      try {
        let vals: ArrayLike<number> | null = null;
        if (isInline) {
          const iv = inlineValues(value);
          vals = iv ? (iv as number[]) : null;
          if (iv) setHead(iv.slice(0, 48));
        } else if (isBuffer && url) {
          if (sizeBytes !== null && sizeBytes > PREVIEW_CAP_BYTES) {
            if (alive) setCapped(true);
            return;
          }
          const t = await fetchTyped(value as never);
          if (!alive) return;
          if (t) {
            vals = t.data as ArrayLike<number>;
            const f32 = new Float32Array(t.data.length);
            for (let i = 0; i < t.data.length; i++) f32[i] = Number(t.data[i]);
            setFlat(f32);
          }
        } else {
          // non-ref value: flatten to numbers for stats + histogram
          const s = await statsOf(value);
          if (s) {
            const flatArr = collect(Array.isArray(value) ? value : value);
            if (flatArr.length) vals = flatArr as number[];
          }
        }
        if (!alive) return;
        if (vals && (vals as ArrayLike<number>).length > 0) setDist(summarize(vals));
      } finally {
        if (alive) setBusy(false);
      }
    })();
    return () => { alive = false; };
  }, [value, isBuffer, isInline, url, sizeBytes]);

  async function handleDownloadNpy() {
    try {
      setDownloading(true);
      const info = dtypeInfo(dtypeName);
      const expectedBytes = flatShape(shapeArr) * info.itemsize;

      let payload: Uint8Array;
      const src: unknown = value;
      if (isBuffer && url) {
        // Large payload: the artifact already holds the raw, dtype-correct
        // little-endian bytes (C-order) — wrap it verbatim, no re-encode.
        const ab = await (await fetch(url)).arrayBuffer();
        payload = new Uint8Array(ab);
      } else if (src && typeof src === "object" && !Array.isArray(src) && "values" in (src as object)) {
        payload = encodeFlat(collect((src as { values: unknown }).values), dtypeName);
      } else if (Array.isArray(src)) {
        payload = encodeFlat(collect(src), dtypeName);
      } else {
        payload = new Uint8Array(0);
      }

      // Sanity: payload byte size should equal elements * itemsize.
      if (expectedBytes > 0 && payload.length !== expectedBytes) {
        console.warn(`[tensor] payload ${payload.length} B != expected ${expectedBytes} B (${dtypeName} ${tupleKey})`);
      }
      saveNpy(payload, dtypeName, shapeArr, npyFileName);
    } finally {
      setDownloading(false);
    }
  }

  // ---------------------------------------------------------- spatial block
  let spatialBlock: ReactNode = null;
  if (spatial === "flow") {
    spatialBlock = (
      <div style={{ marginTop: 10 }}>
        <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-dim)", marginBottom: 4 }}>
          preview · vector field (u, v per pixel)
        </div>
        <FlowField value={value} base={base ?? null} />
      </div>
    );
  } else if (spatial === "heat" && flat && shapeArr.length === 2) {
    spatialBlock = (
      <div style={{ marginTop: 10 }}>
        <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-dim)", marginBottom: 4 }}>
          preview · spatial (H×W)
        </div>
        <HeatPreview values={flat} H={shapeArr[0]} W={shapeArr[1]} />
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-dim)" }}>
          <span>low</span>
          <div style={{ flex: "0 1 180px", height: 8, borderRadius: 4, background: HEAT_GRADIENT }} />
          <span>high</span>
          <span>(viridis, min → max)</span>
        </div>
      </div>
    );
  }

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      <div className="tensorhead">
        <span className="kv"><b>type</b>{kind === "array" ? "inline" : kind ?? "value"}</span>
        <span className="kv"><b>dtype</b>{dtype || "mixed"}</span>
        <span className="kv"><b>shape</b>{shape}</span>
        {size !== null && <span className="kv"><b>size</b>{bytesShort(size)}</span>}
        {(isBuffer || isInline) && (
          <button
            className="btn small"
            disabled={downloading}
            onClick={handleDownloadNpy}
            title={`Download .npy (dtype ${dtypeName}, C-order)`}
          >
            {downloading ? "…" : `↓ .npy (${dtypeName})`}
          </button>
        )}
      </div>
      {capped && (
        <div className="viz-note">
          payload &gt; {bytesShort(PREVIEW_CAP_BYTES)} — stats need the full payload;
          use &ldquo;&darr; .npy&rdquo; to pull it
        </div>
      )}
      {spatialBlock}
      {dist && <DistPlot d={dist} label="value distribution" />}
      {head && head.length > 0 && (
        <pre className="json" style={{ maxHeight: 160, overflow: "auto" }}>
          {head.map((x) => Number(x.toPrecision(5))).join(" ")}{head.length >= 48 ? " …" : ""}
        </pre>
      )}
      {busy && !dist && <span className="spinnerbox"><span className="spinner" /></span>}
    </div>
  );
}
