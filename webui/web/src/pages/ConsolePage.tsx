/** Box console (spec: STATUS.md §4).
 *
 *  Left column  — form generated entirely from the box definition
 *                 (action / command / parameters / section / inputs / session).
 *  Right column — result panel: def-driven visualizers + raw JSON + artifacts
 *                 + call history (re-render on click).
 *
 *  No box knowledge in here: everything comes from ``def`` + the wire
 *  contract in api.ts (backend = core/caller.py build_call, frozen).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api";
import type { BoxDef, CallResult, FleetEntry, ParamDef, SerRef } from "../api";
import { Widget } from "../form/widgets";
import type { FValue } from "../form/widgets";
import { ErrorBox, Spinner, StatusChip, bytesShort, fmtNum } from "../ui";
import { fetchTyped, flattenNumbers, inlineValues, isRef } from "../resolvers";
import { JsonTree, RenderValue } from "../viz/JsonTree";
import { DataTable } from "../viz/DataTable";
import { ImageGrid } from "../viz/ImageGrid";
import { MatrixHeatmap } from "../viz/MatrixHeatmap";
import { FieldMap } from "../viz/FieldMap";
import { FlowField } from "../viz/FlowField";
import { PointCloud } from "../viz/PointCloud";
import { TensorView } from "../viz/TensorView";
import { OverlayViz } from "../viz/OverlayViz";
import { TracksPlayer } from "../viz/TracksPlayer";
import type { TrackStep } from "../viz/TracksPlayer";
import { GLBView } from "../viz/GLBView";

// ----------------------------------------------------------------- helpers

type View =
  | { kind: "res"; res: CallResult }
  | { kind: "err"; detail: Record<string, unknown> };

interface HistItem {
  at: number;                 // Date.now()
  boxName: string;
  command: string | null;
  ok: boolean;
  status: unknown;            // from the box's response config (or null)
  durationMs: number | null;
  images: string[];           // "@token" refs uploaded for this call
  texts: Record<string, string[]>;  // string-list form values (labels for matrices)
  view: View;
}

function detailOf(e: unknown): Record<string, unknown> {
  if (e instanceof ApiError) return { status: e.status, ...e.detail };
  return { message: String(e) };
}

/** "status" / "error" are top-level on the wire (see serialize.py);
 *  ``config_extra`` carries only the box section's *other* keys. */


/** upload ref "@tok" -> fetchable URL; pass through real URLs. */
function fileUrl(ref: string): string {
  return ref.startsWith("@") ? `/api/file/${ref.slice(1)}` : ref;
}

/* tapnext pattern: opaque capability string keying a tracker session. */
function genSession(): string {
  return "ses_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
}

/** shape of a serialized value (ref or plain JSON array). */
function shapeOf(v: unknown): number[] | null {
  if (isRef(v)) return ((v as { shape?: number[] }).shape ?? []).map(Number);
  if (Array.isArray(v)) {
    const out: number[] = [];
    let cur: unknown = v;
    while (Array.isArray(cur)) {
      out.push(cur.length);
      cur = cur.length ? (cur as unknown[])[0] : null;
    }
    return out;
  }
  return null;
}

/** Numeric content of a value: inline array, buffer artifact, or JSON. */
async function numbersOf(v: unknown): Promise<number[] | null> {
  if (isRef(v)) {
    if (v.kind === "array") return inlineValues(v);
    if (v.kind === "buffer") {
      const t = await fetchTyped(v);
      return t ? Array.from(t.data) : null;
    }
  }
  return flattenNumbers(v);
}

// ----------------------------------------------------------------- console

export default function ConsolePage({ defs, defId }: { defs: BoxDef[]; defId: string }) {
  const def = defs.find((d) => d.id === defId);

  if (!def) {
    return (
      <div>
        <h2>unknown box def</h2>
        <p className="sub">no definition <span className="mono">{defId}</span> loaded</p>
        <ErrorBox detail={{ known: defs.map((d) => d.id) }} />
      </div>
    );
  }
  // key by def.id: switching tabs must remount — otherwise actionSel/
  // commandSel/dataVals/fleetId from the *previous* box's console leak into
  // the next one and a "Call" without touching a selector fires the old box's
  // action/command/data.
  return <Console key={def.id} def={def} />;
}

// (split so the hooks below all see a stable `def`)
function Console({ def }: { def: BoxDef }) {
  // ------------------------------------------------------------------ state
  const [entries, setEntries] = useState<FleetEntry[] | null>(null);
  const [fleetId, setFleetId] = useState<string | null>(null);
  const [actionSel, setActionSel] = useState<string | null>(def.actions[0]?.name ?? null);
  const [commandSel, setCommandSel] = useState<string | null>(
    def.command?.default ?? def.command?.values?.[0] ?? null);
  const [dataVals, setDataVals] = useState<Record<string, FValue>>(() => {
    const out: Record<string, FValue> = {};
    for (const f of def.inputs) if (f.default !== undefined) out[f.field] = f.default as FValue;
    return out;
  });
  const [paramVals, setParamVals] = useState<Record<string, FValue>>(() => {
    const out: Record<string, FValue> = {};
    const all = def.parameters;
    for (const p of all) if (p.default !== undefined && p.default !== null) out[p.key] = p.default as FValue;
    return out;
  });
  const [secVals, setSecVals] = useState<Record<string, FValue>>(() => {
    const out: Record<string, FValue> = {};
    for (const s of def.section) if (s.default !== undefined && s.default !== null) out[s.key] = s.default as FValue;
    return out;
  });
  const [session, setSession] = useState("");
  const [calling, setCalling] = useState(false);
  const [view, setView] = useState<View | null>(null);
  const [history, setHistory] = useState<HistItem[]>([]);

  // ------------------------------------------- fleet entries for this def
  useEffect(() => {
    api.fleet()
      .then((r) => {
        const mine = r.entries.filter((e) => e.def_id === def.id || e.name === def.id || e.id === def.id);
        setEntries(mine);
        setFleetId((cur) => cur ?? mine[0]?.id ?? null);
      })
      .catch(() => setEntries([]));
  }, [def.id]);

  // -------------------------------------------------------- session (tapnext)
  // A reload / tab switch remounts this console and the viewer's history is
  // dropped (pure React state).  A *persisted* session_id would then hand the
  // box a live-but-dead session while the viewer is clean: the next call
  // returns tracks spanning the dead frames + the new ones (nonsense).
  // So: fresh box session per (re)mount — viewer and box stay in sync —
  // and `regenerate` restarts mid-sequence the same way.
  const sessionKey = `visionist-webui-session-${def.id}-${fleetId}`;

  useEffect(() => {
    if (!def.session) return;
    setSession(genSession());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [def.session, sessionKey]);

  function regenSession(): void {
    setSession(genSession());
  }

  // -------------------------------------------------- image count for widgets
  const imageField = def.inputs.find((f) => f.widget === "image_upload" || f.widget === "video_frames");
  const imageRefs: string[] = useMemo(() => {
    if (!imageField) return [];
    const v = dataVals[imageField.field];
    return Array.isArray(v) ? v.filter((x) => typeof x === "string") : [];
  }, [imageField, dataVals]);
  const imageCount = imageRefs.length;

  // --------------------------------------------------------------- payloads
  function dataPayload(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const f of def.inputs) {
      const v = dataVals[f.field];
      if (v === undefined || v === null) continue;
      if (typeof v === "string") {
        if (f.widget === "json") { try { out[f.field] = JSON.parse(v); } catch { out[f.field] = v; } continue; }
        if (!v.trim()) continue;
        out[f.field] = v;
        continue;
      }
      if (Array.isArray(v)) {
        const filtered = v.filter((x) => x !== null && x !== undefined && x !== "");
        if (!filtered.length) continue;
        out[f.field] = filtered;
        continue;
      }
      if (typeof v === "number") out[f.field] = v;
    }
    return out;
  }

  function knownParams(): ParamDef[] {
    const out = [...def.parameters];
    const a = def.actions.find((x) => x.name === actionSel);
    if (a) out.push(...a.parameters);
    return out;
  }

  /** string-list form values snapshot (data/params/section) — used later as
   *  axis labels etc. when re-rendering this call. */
  function textSnapshot(): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    const add = (m: Record<string, FValue>): void => {
      for (const [k, v] of Object.entries(m)) {
        if (Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === "string" && !(x as string).startsWith("@"))) out[k] = v as string[];
      }
    };
    add(dataVals); add(paramVals); add(secVals);
    return out;
  }

  function paramPayload(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const p of knownParams()) {
      const v = paramVals[p.key];
      if (v === undefined || v === null || v === "") continue;
      if (typeof v === "string" && p.widget === "json") {
        try { out[p.key] = JSON.parse(v); } catch { /* fall through to literal */ }
      }
      if (!(p.key in out)) out[p.key] = v;
    }
    return out;
  }

  function secPayload(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const s of def.section) {
      const v = secVals[s.key];
      if (v === undefined || v === null || v === "") continue;
      out[s.key] = v;
    }
    return out;
  }

  // -------------------------------------------------------------------- call
  const entryName = (id: string | null): string =>
    entries?.find((e) => e.id === id)?.name ?? id ?? def.id;

  const doCall = useCallback(async (over: { command?: string | null } = {}) => {
    if (!fleetId) return;
    setCalling(true);
    try {
      const command = over.command !== undefined ? over.command : commandSel;
      const res = await api.call({
        fleet_id: fleetId,
        def_id: def.id,
        data: dataPayload(),
        parameters: paramPayload(),
        section: secPayload(),
        command: command ?? null,
        action: actionSel,
        session_id: def.session ? (session || null) : null,
        timeout: 600,
      });
      const st = res.status;
      const okView: View = { kind: "res", res };
      setView(okView);
      setHistory((h) => {
        const entry: HistItem = {
          at: Date.now(), boxName: entryName(fleetId), command,
          ok: true, status: st, durationMs: res.duration_ms,
          images: imageRefs, texts: textSnapshot(), view: okView,
        };
        // A *successful* `reset` clears the box's session — clear the
        // console's accumulated view of it too: the tracks player assembles
        // its steps from this history, so without this it keeps replaying
        // the pre-reset frames (and trails across the reset).  A failed
        // reset keeps history (the box kept its state).
        return command === "reset" && st === "done" ? [entry] : [entry, ...h].slice(0, 10);
      });
    } catch (e) {
      const detail = detailOf(e);
      const errView: View = { kind: "err", detail };
      setView(errView);
      setHistory((h) => [
        {
          at: Date.now(), boxName: entryName(fleetId),
          command: over.command !== undefined ? over.command : commandSel,
          ok: false, status: (detail["status"] as unknown) ?? null,
          durationMs: null, images: imageRefs, texts: textSnapshot(), view: errView,
        },
        ...h,
      ].slice(0, 10));
    } finally {
      setCalling(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    fleetId, def, commandSel, actionSel, def.session, session,
    dataVals, paramVals, secVals, imageRefs, entries,
  ]);

  // ------------------------------------------------------------ form pieces
  const hasReset = !!def.command?.values?.includes("reset");

  return (
    <div>
      <h2>{def.name}</h2>
      {def.experimental && <span className="chip warn" style={{ margin: "0 0 6px" }}>experimental</span>}
      <p className="sub">
        {def.note}
        {def.docs && <span> · <a href={def.docs} target="_blank" rel="noreferrer">box README</a></span>}
      </p>

      <div className="grid-2">
        {/* ------------------------------------------------------ form (L) */}
        <div className="panel">
          <h3>call</h3>

          {!entries && (
            <div className="note" style={{ marginBottom: 10 }}>loading fleet…</div>
          )}
          {entries && entries.length === 0 && (
            <div className="note" style={{ marginBottom: 10 }}>
              no fleet entries for <b>{def.id}</b> yet —{" "}
              <a href="#/fleet">add one on the fleet page</a>, then come back.
            </div>
          )}

          {entries && entries.length > 1 && (
            <label className="fld">
              <span className="lbl">fleet entry</span>
              <select value={fleetId ?? ""} onChange={(e) => setFleetId(e.target.value)}>
                {entries.map((e) => (
                  <option key={e.id} value={e.id}>{e.name} — {e.addr}</option>
                ))}
              </select>
            </label>
          )}
          {entries && entries.length === 1 && (
            <div className="note" style={{ marginBottom: 10 }}>
              → <span className="mono">{entries[0].name} @ {entries[0].addr}</span>
            </div>
          )}

          {def.actions.length > 0 && (
            <label className="fld">
              <span className="lbl">action</span>
              <select value={actionSel ?? ""} onChange={(e) => setActionSel(e.target.value)}>
                {def.actions.map((a) => <option key={a.name} value={a.name}>{a.name}</option>)}
              </select>
            </label>
          )}

          {def.command && def.command.values.length > 0 && (
            <label className="fld">
              <span className="lbl">command</span>
              <select value={commandSel ?? ""} onChange={(e) => setCommandSel(e.target.value)}>
                {def.command.values.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          )}

          {knownParams().length > 0 && (
            <>
              <div className="sect"><div className="sect-title">parameters</div></div>
              {knownParams().map((p) => (
                <label className="fld" key={p.key}>
                  <span className="lbl">{p.key}{p.required && <span className="req">*</span>}</span>
                  <Widget
                    spec={p}
                    value={paramVals[p.key] ?? (p.default as FValue)}
                    onChange={(v) => setParamVals((m) => ({ ...m, [p.key]: v }))}
                    peers={paramVals}
                    imageCount={imageCount}
                  />
                  {p.placeholder && <div className="hint">{p.placeholder}</div>}
                </label>
              ))}
            </>
          )}

          {def.section.length > 0 && (
            <>
              <div className="sect"><div className="sect-title">section</div></div>
              {def.section.map((s) => (
                <label className="fld" key={s.key}>
                  <span className="lbl">{s.key}{s.required && <span className="req">*</span>}</span>
                  <Widget
                    spec={s}
                    value={secVals[s.key] ?? (s.default as FValue)}
                    onChange={(v) => setSecVals((m) => ({ ...m, [s.key]: v }))}
                    peers={secVals}
                    imageCount={imageCount}
                  />
                  {s.placeholder && <div className="hint">{s.placeholder}</div>}
                </label>
              ))}
            </>
          )}

          {def.inputs.length > 0 && (
            <>
              <div className="sect"><div className="sect-title">inputs</div></div>
              {def.inputs.map((f) => (
                <label className="fld" key={f.field}>
                  <span className="lbl">{f.field}{f.required && <span className="req">*</span>}</span>
                  <Widget
                    spec={f}
                    value={dataVals[f.field] ?? (f.default as FValue)}
                    onChange={(v) => setDataVals((m) => ({ ...m, [f.field]: v }))}
                    peers={dataVals}
                    imageCount={imageCount}
                  />
                  {f.helper && <div className="hint">{f.helper}</div>}
                  {f.constraint && <div className="hint">{f.constraint}</div>}
                  {f.placeholder && !f.helper && <div className="hint">{f.placeholder}</div>}
                </label>
              ))}
            </>
          )}

          {def.session && (
            <>
              <div className="sect">
                <div className="sect-title">session (stateful box)</div>
                <button className="btn small" onClick={regenSession}>regenerate</button>
              </div>
              <label className="fld">
                <span className="lbl">{def.session.key}</span>
                <input
                  type="text"
                  className="mono"
                  value={session}
                  onChange={(e) => setSession(e.target.value)}
                />
              </label>
              {def.session.note && <div className="note" style={{ marginBottom: 8 }}>{def.session.note}</div>}
              {def.session.actions.length > 0 && (
                <div className="btnrow">
                  {def.session.actions.map((a) => (
                    <button
                      key={a}
                      className="btn small"
                      disabled={calling || !fleetId}
                      title={`command "${a}" on this session`}
                      onClick={() => void doCall({ command: a })}
                    >
                      {a}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}

          <div className="btnrow" style={{ marginTop: 14 }}>
            <button
              className="btn primary"
              disabled={calling || !fleetId}
              onClick={() => void doCall()}
            >
              {calling ? <Spinner label="calling box…" /> : "Call"}
            </button>
            {hasReset && (
              <button
                className="btn"
                disabled={calling || !fleetId}
                title="explicit reset (this call IS the reset)"
                onClick={() => void doCall({ command: "reset" })}
              >
                Reset
              </button>
            )}
          </div>
        </div>

        {/* ------------------------------------------------------ results (R) */}
        <div>
          {view === null && (
            <div className="panel">
              <div className="empty">
                <div className="big">⚙️</div>
                fill the form and press <b>Call</b> — results render here,
                per the <span className="mono">{def.id}</span> definition.
              </div>
            </div>
          )}

          {view && (
            <ResultView
              def={def}
              view={view}
              imageUrls={imageRefs.map(fileUrl)}
              history={history}
              onPick={(v) => setView(v)}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ result panel

function ResultView({
  def, view, imageUrls, history, onPick,
}: {
  def: BoxDef;
  view: View;
  imageUrls: string[];
  history: HistItem[];
  onPick: (view: View) => void;
}) {
  const [showRaw, setShowRaw] = useState(false);

  if (view.kind === "err") {
    return (
      <div className="panel">
        <h3>call failed</h3>
        <ErrorBox detail={view.detail} />
      </div>
    );
  }

  const res = view.res;
  const status = res.status;
  const err = res.error;
  // the history entry that produced this view (carries the form snapshot)
  const current = history.find((h) => h.view === view);
  const enc = res.declared_encoding === null
    ? null
    : (typeof res.declared_encoding === "string"
      ? res.declared_encoding
      : Object.entries(res.declared_encoding).map(([k, v]) => `${k}:${v}`).join(" "));

  // def-driven visualizers: concrete first, "*" is the fallback renderer.
  // A block with ``only_if_missing: X`` renders only when X is absent from
  // the response (preferred-field fallback — e.g. yolo's annotated grid
  // shows only when the box returned no annotated_video).
  const fields = res.fields ?? {};
  const concrete = def.results.filter((r) => r.field !== "*"
    && !(r.only_if_missing && r.only_if_missing in fields));
  const covered = new Set(concrete.map((r) => r.field));
  const wildcard = def.results.find((r) => r.field === "*");
  const uncovered = Object.keys(res.fields || {});
  const wildcardTracks = wildcard !== undefined && wildcard.visualizer === "tracks_player";

  return (
    <div>
      <div className="panel">
        {/* head */}
        <div className="resulthead">
          <StatusChip status={status} />
          {err && <span className="chip err">{String(err)}</span>}
          {res.runtime !== null && res.runtime !== undefined && (
            <span className="meta">runtime {fmtNum(Number(res.runtime))} s</span>
          )}
          <span className="meta">{Math.round(res.duration_ms)} ms</span>
          {enc !== null && <span className="chip dim">encoding: {enc}</span>}
        </div>

        {Object.keys(res.config_extra || {}).length > 0 && (
          <div style={{ marginBottom: 10 }}>
            {Object.entries(res.config_extra).map(([k, v]) => (
              <div key={k} className="hint" style={{ fontFamily: "var(--mono)", fontSize: 12 }}>
                <span className="k">{k}</span>: <RenderValue v={v} name={k} />
              </div>
            ))}
          </div>
        )}

        {/* input mosaic: what went in (images) — defs whose results already
            show the input (yolo's annotated frames) set input_mosaic: false */}
        {def.input_mosaic !== false && current && current.images.length > 0 && (
          <div className="viz">
            <div className="viz-caption">inputs · {current.images.length} image{current.images.length > 1 ? "s" : ""} uploaded</div>
            <ImageGrid value={current.images.map(fileUrl)} />
          </div>
        )}

        {/* visualizers */}
        {concrete.map((rd, i) => (
          <div className="viz" key={`${rd.field}-${rd.visualizer}-${i}`}>
            <ResultBlock rd={rd} def={def} res={res} baseImages={imageUrls} history={history} current={current} />
          </div>
        ))}

        {wildcard && !wildcardTracks && uncovered
          .filter((f) => !covered.has(f))
          .map((f) => (
            <div className="viz" key={`wcard-${f}`}>
              <div className="viz-caption">{f} (fallback)</div>
              <RenderValue v={res.fields[f]} name={f} />
            </div>
          ))}

        {wildcard && wildcardTracks && (
          <div className="viz">
            <TracksSteps def={def} history={history} />
          </div>
        )}

        {/* artifacts */}
        <details style={{ marginTop: 6 }}>
          <summary style={{ cursor: "pointer", color: "var(--fg-dim)", fontSize: 12.5 }}>
            artifacts ({res.artifacts.length})
          </summary>
          <div style={{ marginTop: 6 }}>
            {res.artifacts.length === 0 && <span className="note">none produced by this call</span>}
            {res.artifacts.map((a) => (
              <span key={a.token} className="artifact">
                <span className="kind">{a.content_type}</span>
                <span>{bytesShort(a.size)}</span>
                {a.extra && a.extra["shape"] !== undefined && (
                  <span className="kind">{String(a.extra["shape"])}</span>
                )}
                <a href={a.url} target="_blank" rel="noreferrer">open</a>
                <a href={a.url} download={a.extra && typeof a.extra["filename"] === "string" ? a.extra["filename"] as string : undefined}>
                  download
                </a>
              </span>
            ))}
          </div>
        </details>

        {/* raw JSON */}
        <div style={{ marginTop: 10 }}>
          <button className="btn small" onClick={() => setShowRaw((s) => !s)}>
            {showRaw ? "hide" : "show"} raw JSON
          </button>
          {showRaw && (
            <div style={{ marginTop: 8 }}>
              <JsonTree data={res.fields} />
            </div>
          )}
        </div>
      </div>

      {/* history */}
      {history.length > 0 && <HistoryPanel history={history} onPick={(h) => onPick(h.view)} />}
    </div>
  );
}

// ------------------------------------------------------- def-driven block

function ResultBlock({
  rd, def, res, baseImages, history, current,
}: {
  rd: BoxDef["results"][number];
  def: BoxDef;
  res: CallResult;
  baseImages: string[];
  history: HistItem[];
  current?: HistItem;
}) {
  const v = res.fields[rd.field];
  const title = rd.caption || rd.field;

  switch (rd.visualizer) {
    case "json":
      return <JsonTree data={v} />;
    case "table":
      return <DataTable value={v} title={title} filename={rd.params?.filename as string || undefined} />;
    case "image_grid":
      return <ImageGrid value={v} title={title} />;
    case "matrix": {
      // axis labels, declared on the result def (params: {row_labels/col_labels: <input field>});
      // params.prop: pick a sub-field of each per-item dict (one small matrix per item)
      const params = rd.params ?? {};
      const labels = (field: unknown): string[] | undefined => {
        if (typeof field !== "string" || !current) return undefined;
        const arr = current.texts[field];
        if (arr && arr.length) return arr;
        const inp = def.inputs.find((f) => f.field === field);
        if (inp && ["image_upload", "video_frames", "file_upload"].includes(inp.widget) && current.images.length) {
          return current.images.map((_, i) => `img ${i + 1}`);
        }
        return undefined;
      };
      return (
        <MatrixHeatmap
          value={v}
          title={title}
          prop={typeof params["prop"] === "string" ? params["prop"] : undefined}
          rowLabels={labels(params["row_labels"])}
          colLabels={labels(params["col_labels"])}
        />
      );
    }
    case "field_map": {
      // per-item numeric map as an image (H×W heat or H×W×3 RGB);
      // params.prop names the item field (box-specific name lives in the def only)
      const prop = (rd.params ?? {})["prop"];
      return (
        <FieldMap
          value={v}
          prop={typeof prop === "string" ? prop : undefined}
          title={title}
        />
      );
    }
    case "flow_field": {
      // dense (u,v per pixel) vector field (e.g. optical flow) -> quiver over
      // the first uploaded input image (def `base`), magnitude-heat toggle.
      const base = rd.base && current && (current.images[0] ?? null)
        ? fileUrl(current.images[0])
        : null;
      return <FlowField value={v} base={base ?? null} title={title} />;
    }
    case "tensor":
      return <TensorView value={v} title={title} />;
    case "overlay": {
      const items = v === undefined || v === null ? [] : Array.isArray(v) ? (v as unknown[]) : [v];
      return (
        <div>
          {title && <div className="viz-caption">{title}</div>}
          <OverlayViz items={items} baseImages={baseImages} layers={rd.layers} />
        </div>
      );
    }
    case "glb": {
      const url = isRef(v) ? (v as { url?: string }).url : typeof v === "string" ? v : null;
      return url
        ? <GLBView url={url} title={title} />
        : <div className="note">no glb payload in this response</div>;
    }
    case "video": {
      // video file artifact (e.g. an annotated mp4) -> native player
      const url = isRef(v) ? (v as { url?: string }).url : typeof v === "string" ? v : null;
      return (
        <div>
          {title && <div className="viz-caption">{title}</div>}
          {url
            ? <video controls preload="metadata" src={url}
                     style={{ maxWidth: "100%", maxHeight: 480, background: "#000" }} />
            : <div className="note">no video payload in this response</div>}
        </div>
      );
    }
    case "points": {
      // per-item point clouds → orbiting three.js scenes; the def names the
      // item props.  Preferred: `depth` (+ `intrinsics`, pixel back-projection);
      // fallback: `points` ((…, 3) positions).  `rd.base` names the input
      // field whose uploaded images color the points — resolved from the
      // per-call snapshot (`current.images`) so history re-renders stay
      // aligned.  Rendered straight from typed arrays — no GLB encoding.
      const p = (rd.params ?? {}) as Record<string, unknown>;
      const s = (x: unknown): string | undefined => (typeof x === "string" ? x : undefined);
      const depth = s(p["depth"]);
      const points = s(p["points"]);
      if (!depth && !points) {
        return (
          <div>
            <div className="viz-caption">{title}</div>
            <div className="note">points needs params.depth + params.intrinsics (reproject from depth) or params.points ((…, 3) positions)</div>
          </div>
        );
      }
      const baseImages = rd.base && current ? current.images.map(fileUrl) : [];
      return (
        <PointCloud
          value={v}
          depth={depth}
          intrinsics={s(p["intrinsics"])}
          projection={s(p["projection"])}
          mask={s(p["mask"])}
          points={points}
          baseImages={baseImages}
          title={title}
        />
      );
    }
    case "download": {
      // A single file ref, or a list of them (e.g. per-frame JPEGs) → one
      // link per item.  No preview: the point is to grab the raw files.
      const fname = rd.params?.filename as string | undefined;
      const refs = Array.isArray(v) && v.length > 0 && v.every(isRef)
        ? (v as SerRef[])
        : isRef(v) ? [v as SerRef] : [];
      return (
        <div>
          <div className="viz-caption">{title}</div>
          {refs.length
            ? refs.map((ref, i) => (
                <span key={i} className="artifact">
                  <span className="kind">{fname || rd.field}{refs.length > 1 ? ` #${i + 1}` : ""}</span>
                  {typeof (ref as any).size === "number" && <span>{bytesShort((ref as any).size)}</span>}
                  <a href={(ref as { url?: string }).url} download={fname || undefined}>{fname || "download"}</a>
                </span>
              ))
            : <div className="note">no file artifact for {rd.field}</div>}
        </div>
      );
    }
    case "tracks_player":
      return <TracksSteps def={def} history={history} title={title} />;
    default:
      return (
        <div>
          <div className="viz-caption">{title} (visualizer “{rd.visualizer}” not implemented)</div>
          <RenderValue v={v} name={rd.field} />
        </div>
      );
  }
}

/** tracks_player steps — assembled from this console's call history.
 *
 *  Each successful entry contributes one step *per uploaded frame*: the
 *  tracks tensor (F, T, 2) index for image ``j`` is ``j + (F - M)`` clamped
 *  to F-1 — so a one-frame call still uses its last (newest) frame, and a
 *  video call (M frames in one call, F == M) maps image j to tracks frame j. */
function TracksSteps({ def, history, title }: { def: BoxDef; history: HistItem[]; title?: string }) {
  const [steps, setSteps] = useState<TrackStep[] | null>(null);

  const tracksField = def.results.find((r) => r.visualizer === "tracks_player")?.inputs["tracks"];
  const visiblesField = def.results.find((r) => r.visualizer === "tracks_player")?.inputs["visibles"];

  useEffect(() => {
    let alive = true;
    void (async () => {
      if (!tracksField) { if (alive) setSteps([]); return; }
      const out: TrackStep[] = [];
      for (const h of [...history].reverse()) {   // oldest -> newest playback order
        if (h.view.kind !== "res") continue;
        const res = h.view.res;
        const tv = res.fields[tracksField];
        if (tv === undefined || tv === null) continue;
        const nums = await numbersOf(tv);
        const shape = shapeOf(tv);
        const time = new Date(h.at).toLocaleTimeString();
        if (nums && shape && shape.length >= 3 && shape[shape.length - 1] === 2) {
          const T = shape[shape.length - 2];
          const F = shape.slice(0, -2).reduce((a, b) => a * b, 1) || 1;
          const M = Math.max(h.images.length, 1);
          const vv = visiblesField ? res.fields[visiblesField] : undefined;
          const vnums = vv === undefined || vv === null ? null : await numbersOf(vv);
          // box replies with (F, T, 1) or (F, T); T is known from tracks
          const V = vnums ? (Math.round(vnums.length / T) || 0) : 0;
          const K = Math.max(1, Math.min(F, M));  // steps for this entry
          for (let j = 0; j < K; j++) {
            const idx = Math.min(j + Math.max(F - K, 0), F - 1);
            const base = idx * T * 2;
            if (nums.length < base + T * 2) continue;
            // box returns (y, x) per its contract — flip to (x, y) for drawing
            const points: { x: number; y: number; visible?: boolean }[] = Array.from({ length: T }, (_, i) => ({
              x: nums[base + i * 2 + 1],
              y: nums[base + i * 2],
            }));
            if (vnums && V >= 1) {
              const vib = Math.min(j + Math.max(V - K, 0), V - 1) * T;
              for (let i = 0; i < T; i++) points[i].visible = !!vnums[vib + i];  // flags may be bool or 0/1
            }
            out.push({
              frameUrl: h.images[j] ? fileUrl(h.images[j]) : null,
              points,
              label: `${h.command ?? ""} ${K > 1 ? `#${j + 1}/${K} ` : ""}${time}`.trim(),
            });
          }
          continue;
        }
        out.push({
          frameUrl: h.images[0] ? fileUrl(h.images[0]) : null,
          points: undefined,
          label: `${h.command ?? ""} ${time}`.trim(),
        });
      }
      if (alive) setSteps(out);
    })();
    return () => { alive = false; };
  }, [history, def, tracksField, visiblesField]);

  if (steps === null) return <Spinner label="assembling tracking steps…" />;
  return <TracksPlayer steps={steps} title={title || "tracks (session history)"} />;
}

// ----------------------------------------------------------------- history

function HistoryPanel({
  history, onPick,
}: {
  history: HistItem[];
  onPick: (h: HistItem) => void;
}) {
  return (
    <div className="panel">
      <h3>call history (last {history.length})</h3>
      <div className="hist">
        {history.map((h, i) => (
          <div
            key={h.at + "-" + i}
            className="item"
            style={{ cursor: "pointer" }}
            title="click to re-render this response"
            onClick={() => onPick(h)}
          >
            <StatusChip status={h.ok ? h.status : "error"} />
            <span className="grow mono" style={{ color: "var(--fg-dim)" }}>
              {new Date(h.at).toLocaleTimeString()} · {h.boxName}
              {h.command ? ` · ${h.command}` : ""}
            </span>
            <span className="mono" style={{ color: "var(--fg-dim)" }}>
              {h.durationMs !== null ? `${Math.round(h.durationMs)} ms` : "—"}
            </span>
          </div>
        ))}
      </div>
      <div className="note">click an entry to re-render that response above.</div>
    </div>
  );
}
