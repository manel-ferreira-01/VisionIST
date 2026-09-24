/** flow_field visualizer — render a *dense 2-D vector field* (a per-pixel
 *  (u, v) displacement; e.g. optical flow) as a quiver of arrows over the
 *  first base image, or as a per-pixel magnitude heat image.  Box-agnostic:
 *  the def points a result field at the numeric (… , 2, H, W) field and may
 *  name the base image via `base`.
 *
 *  Accepted wire shapes (float buffer/array):
 *    (N, 2, H, W)  N vector fields (e.g. fwd[, bwd]); shows field 0 and a
 *                  selectable index when N > 1
 *    (2, H, W)     a single vector field
 *    (H, W, 2)     a single vector field, last-axis components
 *    (H, W)        scalar magnitude (heat only, no arrows)
 *
 *  Arrows point in the (u, v) direction (screen coords: +u right, +v down)
 *  with length ∝ magnitude (clamped to the cell so the field stays
 *  readable); the palette (viridis) matches every other heatmap in the SPA.
 *  Degrades to metadata, never breaks.
 */
import { useEffect, useRef, useState } from "react";
import { fetchTyped } from "../resolvers";

const HEAT_GRADIENT =
  "linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))";

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

const MAX_DRAW = 1024;   // render resolution cap (source grid stays full res)

type Interpret =
  | { kind: "vector"; n: number; H: number; W: number; layout: "chn" | "hwc" }
  | { kind: "scalar"; H: number; W: number }
  | { kind: "meta" };

function interpret(shape: number[]): Interpret {
  const s = shape.filter((x) => x > 0);
  if (s.length === 4 && s[1] === 2) return { kind: "vector", n: s[0], H: s[2], W: s[3], layout: "chn" };
  if (s.length === 3 && s[0] === 2) return { kind: "vector", n: 1, H: s[1], W: s[2], layout: "chn" };
  if (s.length === 3 && s[s.length - 1] === 2) return { kind: "vector", n: 1, H: s[0], W: s[1], layout: "hwc" };
  if (s.length === 2) return { kind: "scalar", H: s[0], W: s[1] };
  return { kind: "meta" };
}

export function FlowField({ value, base, title }: {
  value: unknown;
  base?: string | null;
  title?: string;
}) {
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<"quiver" | "heat">("quiver");
  const [fieldIdx, setFieldIdx] = useState(0);
  const [interp, setInterp] = useState<Interpret>({ kind: "meta" });
  const [stats, setStats] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<Float32Array | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const ref = useRef<HTMLCanvasElement | null>(null);

  // ------------------------------------------------------------- fetch data
  useEffect(() => {
    let alive = true;
    setReady(false); setErr(null); setData(null); setUrl(null);
    void (async () => {
      try {
        const t = await fetchTyped(value as never);
        if (!alive) return;
        if (!t || t.data.length === 0) { setErr("no numeric “flow” field in the response"); return; }
        const d = t.data as Float32Array;
        setData(d);
        setUrl(("url" in (value as object) && typeof (value as { url?: string }).url === "string") ? (value as { url: string }).url : null);
        const ip = interpret(t.shape);
        setInterp(ip);
      } catch (e) { if (alive) setErr(String(e)); }
    })();
    return () => { alive = false; };
  }, [value]);

  // --------------------------------------------------------------- draw grid
  useEffect(() => {
    if (interp.kind === "meta" || !data) { setReady(true); return; }
    const { H, W } = interp as { H: number; W: number };
    if (!H || !W || data.length < H * W) { setErr(`shape/data mismatch (${H}×${W})`); setReady(true); return; }

    const scale = Math.min(1, MAX_DRAW / Math.max(H, W));
    const cw = Math.max(2, Math.round(W * scale));
    const chh = Math.max(2, Math.round(H * scale));
    const canvas = ref.current;
    if (!canvas) { setReady(true); return; }
    canvas.width = cw; canvas.height = chh;
    const ctx = canvas.getContext("2d");
    if (!ctx) { setReady(true); return; }

    // value at (field k, component c, row y, col x) -> flat index
    const at = (k: number, c: number, y: number, x: number): number => {
      if (interp.kind === "scalar") return 0;
      if ((interp as { layout: string }).layout === "hwc") return data[(y * W + x) * 2 + c];
      return data[((k * 2 + c) * H + y) * W + x];
    };

    // ---- magnitude range (finite values only) ----------------------------
    let mn = Infinity, mx = -Infinity;
    const magAt = (y: number, x: number, k: number): number => {
      if (interp.kind === "scalar") { const v = data[y * W + x]; return Number.isFinite(v) ? v : NaN; }
      const u = at(k, 0, y, x), v = at(k, 1, y, x);
      if (!Number.isFinite(u) || !Number.isFinite(v)) return NaN;
      return Math.hypot(u, v);
    };
    for (let y = 0; y < H; y += 4)
      for (let x = 0; x < W; x += 4) {
        const m = magAt(y, x, fieldIdx);
        if (!Number.isFinite(m)) continue;
        if (m < mn) mn = m; if (m > mx) mx = m;
      }
    if (!Number.isFinite(mn)) { mn = 0; mx = 1; }
    const span = (mx - mn) || 1;
    const norm = (m: number): number => (m - mn) / span;

    // ---- base image (background) -----------------------------------------
    const paintBase = (img: HTMLImageElement | null) => {
      ctx.clearRect(0, 0, cw, chh);
      if (img) {
        // cover
        const ir = img.width / img.height, cr = cw / chh;
        let dw = cw, dh = chh, dx = 0, dy = 0;
        if (ir > cr) { dw = chh * ir; dx = (cw - dw) / 2; }
        else { dh = cw / ir; dy = (chh - dh) / 2; }
        ctx.globalAlpha = 0.72;
        ctx.drawImage(img, dx, dy, dw, dh);
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = "#0b0f14"; ctx.fillRect(0, 0, cw, chh);
      }
    };

    // ---- mode: heat (scalar magnitude per pixel) -------------------------
    if (mode === "heat" || interp.kind === "scalar") {
      const off = document.createElement("canvas");
      off.width = W; off.height = H;
      const octx = off.getContext("2d");
      if (octx) {
        const im = octx.createImageData(W, H);
        for (let i = 0; i < H * W; i++) {
          const m = interp.kind === "scalar" ? Number(data[i]) : magAt(Math.floor(i / W), i % W, fieldIdx);
          const [r, g, b] = Number.isFinite(m) ? heatRgb(norm(m)) : [16, 21, 28];
          im.data[i * 4 + 0] = r; im.data[i * 4 + 1] = g; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = 255;
        }
        octx.putImageData(im, 0, 0);
        paintBase(null);
        ctx.drawImage(off, 0, 0, cw, chh);
      }
      setStats(`${W} × ${H} · ${interp.kind === "vector" ? (interp as { n: number }).n + " field(s) · " : ""}mag min ${mn.toPrecision(3)} · max ${mx.toPrecision(3)} · ${interp.kind === "vector" ? "(u,v) px" : "scalar"}`);
      setReady(true);
      return;
    }

    // ---- mode: quiver over base ------------------------------------------
    const doQuiver = (img: HTMLImageElement | null) => {
      paintBase(img);
      // adaptive grid so arrow density tracks the image
      const target = 34;
      const step = Math.max(3, Math.round(Math.min(cw, chh) / target));
      let nArrows = 0;
      for (let gy = 0; gy < H; gy += Math.max(1, Math.round(step / scale))) {
        for (let gx = 0; gx < W; gx += Math.max(1, Math.round(step / scale))) {
          const sx = gx * scale, sy = gy * scale;       // arrow origin (display px)
          const m = magAt(gy, gx, fieldIdx);
          if (!Number.isFinite(m) || m === 0) continue;
          const u = at(fieldIdx, 0, gy, gx), v = at(fieldIdx, 1, gy, gx);
          let ex = sx + (u / m) * (m * scale * 6);      // dir * len
          let ey = sy + (v / m) * (m * scale * 6);
          const len = Math.hypot(ex - sx, ey - sy);
          const maxLen = step * 0.95;                    // clamp to the cell
          if (len > maxLen) { ex = sx + (ex - sx) * maxLen / len; ey = sy + (ey - sy) * maxLen / len; }
          const [r, g, b] = heatRgb(norm(m));
          ctx.strokeStyle = `rgba(${r | 0},${g | 0},${b | 0},0.95)`;
          ctx.fillStyle = ctx.strokeStyle;
          ctx.lineWidth = 1.1;
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
          // tiny head
          ctx.beginPath(); ctx.arc(ex, ey, 1.1, 0, Math.PI * 2); ctx.fill();
          nArrows++;
        }
      }
      setStats(`${W} × ${H} · ${nArrows} arrows · mag min ${mn.toPrecision(3)} · max ${mx.toPrecision(3)} px`);
    };

    if (base) {
      const img = new Image();
      img.onload = () => { doQuiver(img); };
      img.onerror = () => { doQuiver(null); };
      img.src = base;
    } else {
      doQuiver(null);
    }
  }, [data, interp, mode, fieldIdx, base]);

  const showFields = interp.kind === "vector" && (interp as { n: number }).n > 1;

  return (
    <div>
      <div className="viz-caption" style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
        <span>{title}</span>
        {showFields && (
          <span style={{ display: "inline-flex", gap: 4 }}>
            {Array.from({ length: (interp as { n: number }).n }, (_, i) => (
              <button key={i} className="btn small"
                style={i === fieldIdx ? { borderColor: "var(--accent)", color: "var(--accent)" } : {}}
                onClick={() => setFieldIdx(i)}>{i === 0 ? "fwd" : `field ${i}`}</button>
            ))}
          </span>
        )}
        {interp.kind === "vector" && (
          <span style={{ display: "inline-flex", gap: 4 }}>
            <button className="btn small" style={mode === "quiver" ? { borderColor: "var(--accent)", color: "var(--accent)" } : {}} onClick={() => setMode("quiver")}>quiver</button>
            <button className="btn small" style={mode === "heat" ? { borderColor: "var(--accent)", color: "var(--accent)" } : {}} onClick={() => setMode("heat")}>magnitude</button>
          </span>
        )}
      </div>
      {err && <div className="note">{err}</div>}
      <canvas ref={(el) => { ref.current = el; }}
        style={{ maxWidth: "100%", display: "block", marginTop: 8, visibility: ready ? "visible" : "hidden", border: "1px solid var(--border)", borderRadius: 6 }} />
      {ready && stats && <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11, marginTop: 6 }}>{stats}</div>}
      {mode === "heat" || interp.kind === "scalar" ? (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontFamily: "var(--mono)", fontSize: 11, color: "var(--fg-dim)" }}>
          <span>low</span>
          <div style={{ flex: "0 1 180px", height: 8, borderRadius: 4, background: HEAT_GRADIENT }} />
          <span>high</span>
          <span>(viridis, magnitude min → max)</span>
        </div>
      ) : (
        interp.kind === "vector" && <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11, marginTop: 6, color: "var(--fg-dim)" }}>
          arrow direction = motion; color + length = magnitude (px)
        </div>
      )}
      {url && <a className="hint" href={url} download style={{ display: "inline-block", marginTop: 6, fontFamily: "var(--mono)", fontSize: 11 }}>download raw</a>}
      {!ready && !err && <span className="spinnerbox"><span className="spinner" /></span>}
    </div>
  );
}
