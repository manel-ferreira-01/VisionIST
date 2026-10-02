/** Definition-driven form widgets.  Each widget renders one InputField /
 *  ParamDef / SectionField spec — no box knowledge in here. */
import { useRef, useState } from "react";
import type { InputField, ParamDef } from "../api";
import { api, ApiError } from "../api";
import { Spinner, bytesShort } from "../ui";

export type FValue =
  | string
  | number
  | string[]          // tags / upload refs / text_repeat
  | null;

interface WidgetProps {
  spec: InputField | ParamDef;
  value: FValue;
  onChange: (v: FValue) => void;
  /** sibling field values (e.g. text_repeat sizes to the image field) */
  peers: Record<string, FValue>;
  imageCount: number;
}

export function Widget(props: WidgetProps) {
  const { spec, value, onChange, imageCount } = props;
  switch (spec.widget) {
    case "image_upload":
      return (
        <UploadWidget
          multiple
          value={(value as string[]) || []}
          onChange={onChange}
          accept="image/*"
        />
      );
    case "video_frames":
      return <VideoFramesWidget value={(value as string[]) || []} onChange={onChange} />;
    case "file_upload":
      return (
        <UploadWidget
          multiple={false}
          value={(value as string[]) || []}
          onChange={onChange}
          accept="*/*"
        />
      );
    case "tags":
      return <TagsWidget value={(value as string[]) || []} onChange={onChange} placeholder={(spec as InputField).placeholder || "type and press Enter"} />;
    case "text_repeat":
      return (
        <TextRepeatWidget
          value={(value as string[]) || []}
          onChange={onChange}
          count={imageCount}
        />
      );    case "slider":
      return (
        <SliderWidget
          min={(spec as ParamDef).min ?? 0} max={(spec as ParamDef).max ?? 1} step={(spec as ParamDef).step ?? 0.01}
          value={typeof value === "number" ? value : (spec.default as number) ?? 0}
          onChange={onChange}
        />
      );
    case "select":
      return (
        <SelectWidget
          options={(spec as ParamDef).values || []}
          value={typeof value === "string" ? value : (spec.default as string) ?? ""}
          placeholder={(spec as ParamDef).placeholder || undefined}
          onChange={onChange}
        />
      );
    case "number":
      return (
        <NumberWidget
          value={typeof value === "number" ? value : value === null || value === undefined ? (spec.default as number) ?? 0 : Number(value)}
          onChange={onChange}
          min={(spec as ParamDef).min ?? undefined} max={(spec as ParamDef).max ?? undefined} step={(spec as ParamDef).step ?? undefined}
        />
      );
    case "json":
      return <JsonWidget value={typeof value === "string" ? value : JSON.stringify(value ?? "", null, 2)} onChange={onChange} />;
    default:
      return <div className="note">unknown widget “{spec.widget}”</div>;
  }
}

// ------------------------------------------------------------------ upload

/** Extract ``n`` evenly spaced frames from a video file in the browser —
 *  over the whole clip, or over ``[from, to]`` seconds — and upload each as a
 *  JPEG image ref ("@token").  Wire shape ends up identical
 *  to an ``image_upload`` multiple field — boxes that accept image lists can
 *  consume video without any backend knowledge of video. */
async function videoToFrameRefs(
  file: File, n: number, onProg?: (done: number) => void,
  range: { from?: number | null; to?: number | null } = {},
): Promise<string[]> {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.preload = "auto";
  v.muted = true;
  (v as HTMLVideoElement & { playsInline?: boolean }).playsInline = true;
  v.src = url;
  try {
    await new Promise<void>((res, rej) => {
      v.onloadeddata = () => res();
      v.onerror = () => rej(new Error("could not decode this video"));
    });
    const dur = isFinite(v.duration) ? v.duration : 0;
    if (!dur) throw new Error("video has no playable duration");
    const t0 = Math.min(Math.max(0, range.from ?? 0), dur);
    const t1 = Math.min(dur, Math.max(t0, range.to ?? dur));
    if (t1 - t0 <= 0) throw new Error(`empty time range ${t0}–${t1} s (video is ${dur.toFixed(2)} s)`);

    const maxDim = 960;
    const scale = Math.min(1, maxDim / Math.max(v.videoWidth, v.videoHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(2, Math.round(v.videoWidth * scale));
    canvas.height = Math.max(2, Math.round(v.videoHeight * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas 2d context unavailable");

    const out: string[] = [];
    for (let i = 0; i < n; i++) {
      const t = t0 + (t1 - t0) * (i + 0.5) / n;   // midpoint sampling: robust for short clips
      await new Promise<void>((res) => {
        v.onseeked = () => res();
        v.currentTime = Math.min(t, Math.max(0, dur - 0.05));
      });
      ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
      const blob: Blob = await new Promise((res, rej) =>
        canvas.toBlob((b) => (b ? res(b) : rej(new Error("frame encode failed"))), "image/jpeg", 0.9));
      const up = await api.upload(new File([blob], `frame_${String(i + 1).padStart(3, "0")}.jpg`, { type: "image/jpeg" }));
      out.push(up.ref);
      onProg?.(i + 1);
    }
    return out;
  } finally {
    v.removeAttribute("src");
    URL.revokeObjectURL(url);
  }
}

function VideoFramesWidget({
  value, onChange,
}: { value: string[]; onChange: (v: string[]) => void }) {
  const [frameCount, setFrameCount] = useState(32);
  const [from, setFrom] = useState<string>("");
  const [to, setTo] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [names, setNames] = useState<Record<string, string>>({});
  const inputRef = useRef<HTMLInputElement>(null);

  async function addFiles(list: FileList | null) {
    if (!list || !list.length || busy) return;
    setBusy(true);
    setErr(null);
    try {
      const next = [...value];
      for (const f of Array.from(list)) {
        if (f.type.startsWith("image/")) {
          const up = await api.upload(f);
          next.push(up.ref);
          setNames((m) => ({ ...m, [up.ref]: `${f.name} (${bytesShort(f.size)})` }));
        } else {
          setProg(0);
          const sec = (x: string): number | null => (x.trim() === "" || !Number.isFinite(Number(x)) ? null : Number(x));
          const n = Math.max(2, Math.min(256, Math.round(frameCount) || 2));
          for (const ref of await videoToFrameRefs(f, n, (d) => setProg(d), { from: sec(from), to: sec(to) })) {
            next.push(ref);
            setNames((m) => ({ ...m, [ref]: `${f.name} · frame ${next.length}` }));
          }
        }
      }
      onChange(next);
    } catch (e) {
      setErr(e instanceof ApiError ? String(e.detail["message"] ?? e.message) : String((e as Error).message));
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center", marginBottom: 6, whiteSpace: "nowrap" }}>
        <span className="hint">video →</span>
        <input
          type="number" min={2} max={256} step={1}
          value={frameCount}
          onChange={(e) => setFrameCount(Number(e.target.value))}
          disabled={busy}
          title="frames to extract from the video (evenly spaced, 2–256)"
          style={{ width: 64 }}
        />
        <span className="hint">frames, from</span>
        <input type="number" min={0} step={0.1} value={from} placeholder="0" disabled={busy}
          onChange={(e) => setFrom(e.target.value)} style={{ width: 64 }} title="start (s); empty = video start" />
        <span className="hint">to</span>
        <input type="number" min={0} step={0.1} value={to} placeholder="end" disabled={busy}
          onChange={(e) => setTo(e.target.value)} style={{ width: 64 }} title="end (s); empty = video end" />
        <span className="hint">s</span>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/mp4,video/webm,video/quicktime,video/x-m4v"
        multiple
        onChange={(e) => void addFiles(e.target.files)}
      />
      {value.length > 0 && (
        <div className="filechips">
          {value.map((ref, i) => (
            <span key={ref + i} className="filechip">
              <span className="mono">{i + 1}.</span> {names[ref] || ref.slice(-8)}
              <button title="remove" onClick={() => onChange(value.filter((_, j) => j !== i))}>×</button>
            </span>
          ))}
        </div>
      )}
      {busy && <div className="note"><Spinner label={prog ? `extracting/uploading ${prog} frames…` : "extracting video…"} /></div>}
      {err && <div className="note" style={{ color: "var(--err)" }}>{err}</div>}
      <div className="hint">video (mp4/webm/mov) or images — video frames are extracted in your browser, sent in order as one image sequence</div>
    </>
  );
}

function UploadWidget({
  multiple, value, onChange, accept,
}: {
  multiple: boolean;
  value: string[];       // "@token" refs, in order
  onChange: (v: string[]) => void;
  accept?: string;
}) {
  const [busy, setBusy] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [names, setNames] = useState<Record<string, string>>({});

  async function addFiles(list: FileList | null) {
    if (!list || !list.length) return;
    setBusy(list.length);
    setErr(null);
    try {
      const next = [...value];
      for (const f of Array.from(list)) {
        const up = await api.upload(f);
        next.push(up.ref);
        setNames((m) => ({ ...m, [up.ref]: `${f.name} (${bytesShort(f.size)})` }));
        setBusy((n) => n - 1);
      }
      onChange(multiple ? next : next.slice(-1));
    } catch (e) {
      setErr(e instanceof ApiError ? String(e.detail["message"] ?? e.message) : String(e));
    } finally {
      setBusy(0);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={(e) => addFiles(e.target.files)}
      />
      {value.length > 0 && (
        <div className="filechips">
          {value.map((ref, i) => (
            <span key={ref + i} className="filechip">
              <span className="mono">{i + 1}.</span> {names[ref] || ref.slice(-8)}
              <button title="remove" onClick={() => onChange(value.filter((_, j) => j !== i))}>×</button>
            </span>
          ))}
        </div>
      )}
      {busy > 0 && <div className="note"><Spinner label={`uploading ${busy}…`} /></div>}
      {err && <div className="note" style={{ color: "var(--err)" }}>{String(err)}</div>}
    </>
  );
}

// --------------------------------------------------------------------- tags

function TagsWidget({
  value, onChange, placeholder,
}: { value: string[]; onChange: (v: string[]) => void; placeholder?: string }) {
  const [draft, setDraft] = useState("");
  return (
    <>
      <input
        type="text"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            const t = draft.trim();
            if (t) onChange([...value, t]);
            setDraft("");
          }
        }}
      />
      {value.length > 0 && (
        <div className="tagchips" style={{ marginTop: 8 }}>
          {value.map((t, i) => (
            <span key={t + i} className="tagchip">
              {t}
              <button onClick={() => onChange(value.filter((_, j) => j !== i))}>×</button>
            </span>
          ))}
        </div>
      )}
    </>
  );
}

// --------------------------------------------------------------- text_repeat

function TextRepeatWidget({
  value, onChange, count,
}: { value: string[]; onChange: (v: string[]) => void; count: number }) {
  const n = Math.max(1, count);
  const val: string[] = Array.from({ length: n }, (_, i) => value[i] ?? value[0] ?? "");
  function set(i: number, v: string) {
    const next = [...val];
    next[i] = v;
    onChange(next);
  }
  if (n === 1) {
    return <input type="text" value={val[0]} onChange={(e) => set(0, e.target.value)} placeholder="one text" />;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {val.map((v, i) => (
        <input key={i} type="text" value={v} placeholder={`text for image ${i + 1}`}
          onChange={(e) => set(i, e.target.value)} />
      ))}
      <button
        className="btn small"
        type="button"
        onClick={() => onChange(val.map(() => val[0]))}
      >
        copy first to all
      </button>
    </div>
  );
}

// ------------------------------------------------------------------- slider

function SliderWidget({
  min, max, step, value, onChange,
}: { min: number; max: number; step: number; value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <div className="sliderrow">
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(Number(e.target.value))} />
        <input type="number" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(Number(e.target.value))} />
      </div>
      <div className="hint">{min} … {max} (step {step})</div>
    </div>
  );
}

// ------------------------------------------------------------------- select

function SelectWidget({
  options, value, onChange, placeholder,
}: { options: string[]; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {placeholder && value === "" && (
        <option value="">{placeholder}</option>
      )}
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

// -------------------------------------------------------------------- number

function NumberWidget({
  value, onChange, min, max, step,
}: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number }) {
  return (
    <input type="number" value={value} min={min} max={max} step={step}
      onChange={(e) => onChange(Number(e.target.value))} />
  );
}

// --------------------------------------------------------------------- json

function JsonWidget({
  value, onChange,
}: { value: string; onChange: (v: string) => void }) {
  const [bad, setBad] = useState<string | null>(null);
  return (
    <>
      <textarea
        className="json"
        value={value}
        spellCheck={false}
        onChange={(e) => {
          onChange(e.target.value);
          try { JSON.parse(e.target.value); setBad(null); }
          catch (err) { setBad(String((err as Error).message)); }
        }}
        onBlur={() => { try { JSON.parse(value); setBad(null); } catch { /* keep */ } }}
      />
      {bad && <div className="note" style={{ color: "var(--err)" }}>invalid JSON: {bad}</div>}
    </>
  );
}
