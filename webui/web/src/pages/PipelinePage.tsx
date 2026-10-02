/** Pipeline page: a def-driven form (same widgets as a box console), a Run
 *  that starts a server-side job, and live progress — steps with their box
 *  calls, then each result block as soon as the pipeline emits its field.
 *
 *  No pipeline knowledge in here: everything comes from the PipelineDef and
 *  the job snapshot (webui/core/pipeline.py).  Result blocks are the box
 *  console's (ResultBlock), so every visualizer works here too. */
import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api";
import type { Job, JobStep, PipelineDef } from "../api";
import { Widget } from "../form/widgets";
import type { FValue } from "../form/widgets";
import { ErrorBox, Spinner, StatusChip, bytesShort, fmtNum } from "../ui";
import { RenderValue } from "../viz/JsonTree";
import { ImageGrid } from "../viz/ImageGrid";
import { ResultBlock, fileUrl } from "./ConsolePage";
import type { HistItem } from "./ConsolePage";

const POLL_MS = 700;
const TERMINAL = new Set(["done", "error", "cancelled"]);

function detailOf(e: unknown): Record<string, unknown> {
  if (e instanceof ApiError) return { status: e.status, ...e.detail };
  return { message: String(e) };
}

/** Keep the previous object for every unchanged field, so visualizers
 *  (which refetch/rebuild on a new ``value``) don't redo work every poll. */
function mergeFields(prev: Record<string, unknown>, next: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  let same = Object.keys(prev).length === Object.keys(next).length;
  for (const [k, v] of Object.entries(next)) {
    if (k in prev && JSON.stringify(prev[k]) === JSON.stringify(v)) out[k] = prev[k];
    else { out[k] = v; same = false; }
  }
  return same ? prev : out;
}

function storeKey(id: string): string {
  return `visionist-webui-job-${id}`;
}

export default function PipelinePage({ pipelines, id }: { pipelines: PipelineDef[]; id: string }) {
  const def = pipelines.find((p) => p.id === id);
  if (!def) {
    return (
      <div>
        <h2>unknown pipeline</h2>
        <ErrorBox detail={{ known: pipelines.map((p) => p.id) }} />
      </div>
    );
  }
  return <Pipeline key={def.id} def={def} />;
}

function Pipeline({ def }: { def: PipelineDef }) {
  const [dataVals, setDataVals] = useState<Record<string, FValue>>(() => {
    const out: Record<string, FValue> = {};
    for (const f of def.inputs) if (f.default !== undefined && f.default !== null) out[f.field] = f.default as FValue;
    return out;
  });
  const [paramVals, setParamVals] = useState<Record<string, FValue>>(() => {
    const out: Record<string, FValue> = {};
    for (const p of def.parameters) if (p.default !== undefined && p.default !== null) out[p.key] = p.default as FValue;
    return out;
  });
  const [missing, setMissing] = useState<string[]>(def.missing);
  const [job, setJob] = useState<Job | null>(null);
  const [fields, setFields] = useState<Record<string, unknown>>({});
  const [startErr, setStartErr] = useState<Record<string, unknown> | null>(null);
  const [starting, setStarting] = useState(false);
  const jobId = useRef<string | null>(null);

  // fleet may have changed since the app loaded the defs
  useEffect(() => {
    api.pipelines()
      .then((r) => setMissing(r.pipelines.find((p) => p.id === def.id)?.missing ?? []))
      .catch(() => { /* keep the listing's */ });
  }, [def.id]);

  // ---------------------------------------------------------------- polling
  function track(id: string): void {
    jobId.current = id;
    try { sessionStorage.setItem(storeKey(def.id), id); } catch { /* private mode */ }
  }

  useEffect(() => {
    // resume the last job of this pipeline after a reload / tab switch
    let saved: string | null = null;
    try { saved = sessionStorage.getItem(storeKey(def.id)); } catch { /* ignore */ }
    if (saved) jobId.current = saved;

    let alive = true;
    let timer = 0;
    const tick = async () => {
      const id = jobId.current;
      if (id) {
        try {
          const j = await api.job(id);
          if (!alive) return;
          if (jobId.current === id) {
            setJob(j);
            setFields((prev) => mergeFields(prev, j.fields));
          }
          if (TERMINAL.has(j.status) && jobId.current === id) jobId.current = null;
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) {
            jobId.current = null;                 // expired / server restarted
            try { sessionStorage.removeItem(storeKey(def.id)); } catch { /* ignore */ }
          }
        }
      }
      if (alive) timer = window.setTimeout(() => void tick(), POLL_MS);
    };
    void tick();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [def.id]);

  // ---------------------------------------------------------------- actions
  function payload(): { data: Record<string, unknown>; parameters: Record<string, unknown> } {
    const data: Record<string, unknown> = {};
    for (const f of def.inputs) {
      const v = dataVals[f.field];
      if (v === undefined || v === null || v === "") continue;
      if (Array.isArray(v)) { if (v.length) data[f.field] = v; continue; }
      data[f.field] = v;
    }
    const parameters: Record<string, unknown> = {};
    for (const p of def.parameters) {
      const v = paramVals[p.key];
      if (v === undefined || v === null || v === "") continue;
      if (typeof v === "string" && p.widget === "json") {
        try { parameters[p.key] = JSON.parse(v); continue; } catch { /* literal */ }
      }
      parameters[p.key] = v;
    }
    return { data, parameters };
  }

  async function run(): Promise<void> {
    setStarting(true);
    setStartErr(null);
    try {
      const j = await api.runPipeline(def.id, { ...payload(), timeout: 600 });
      setFields({});
      setJob(j);
      track(j.id);
    } catch (e) {
      setStartErr(detailOf(e));
    } finally {
      setStarting(false);
    }
  }

  async function cancel(): Promise<void> {
    if (!job) return;
    try { setJob(await api.cancelJob(job.id)); } catch (e) { setStartErr(detailOf(e)); }
  }

  // ---------------------------------------------------------------- derived
  const imageField = def.inputs.find((f) => f.widget === "image_upload" || f.widget === "video_frames");
  const formImages = imageField && Array.isArray(dataVals[imageField.field])
    ? (dataVals[imageField.field] as string[]) : [];
  const running = !!job && !TERMINAL.has(job.status);

  // the job's own inputs (not the form's: the form may have moved on)
  const jobImages: string[] = useMemo(() => {
    const v = imageField && job ? job.inputs[imageField.field] : undefined;
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, imageField?.field]);
  const current: HistItem | undefined = useMemo(() => job ? {
    at: job.created * 1000, boxName: def.id, command: null, ok: true, status: job.status,
    durationMs: null, images: jobImages, texts: {},
    view: { kind: "err", detail: {} },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  } : undefined, [job?.id, jobImages]);
  const baseImages = useMemo(() => jobImages.map(fileUrl), [jobImages]);

  const concrete = def.results.filter((r) => r.field !== "*");
  const covered = new Set(concrete.map((r) => r.field));
  const wildcard = def.results.find((r) => r.field === "*");

  return (
    <div>
      <h2>{def.name}</h2>
      {def.experimental && <span className="chip warn" style={{ margin: "0 0 6px" }}>experimental</span>}
      <p className="sub">
        {def.note}
        {def.docs && <span> · <a href={def.docs} target="_blank" rel="noreferrer">README</a></span>}
      </p>

      <div className="grid-2">
        {/* ------------------------------------------------------ form (L) */}
        <div className="panel">
          <h3>run</h3>
          <div className="note" style={{ marginBottom: 10 }}>
            pipeline over{" "}
            {def.uses.map((u, i) => (
              <span key={u}>
                {i > 0 && " → "}
                <a href={`#/box/${u}`} className="mono"
                  style={missing.includes(u) ? { color: "var(--err)" } : undefined}
                  title={missing.includes(u) ? "no fleet entry serves this box" : "open the box console"}>
                  {u}
                </a>
              </span>
            ))}
            {missing.length > 0 && (
              <div style={{ marginTop: 6 }}>
                no fleet entry for <b>{missing.join(", ")}</b> — <a href="#/fleet">add on the fleet page</a>
              </div>
            )}
          </div>

          {def.parameters.length > 0 && (
            <>
              <div className="sect"><div className="sect-title">parameters</div></div>
              {def.parameters.map((p) => (
                <label className="fld" key={p.key}>
                  <span className="lbl">{p.key}{p.required && <span className="req">*</span>}</span>
                  <Widget
                    spec={p}
                    value={paramVals[p.key] ?? (p.default as FValue)}
                    onChange={(v) => setParamVals((m) => ({ ...m, [p.key]: v }))}
                    peers={paramVals}
                    imageCount={formImages.length}
                  />
                  {p.placeholder && <div className="hint">{p.placeholder}</div>}
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
                    imageCount={formImages.length}
                  />
                  {f.helper && <div className="hint">{f.helper}</div>}
                  {f.constraint && <div className="hint">{f.constraint}</div>}
                </label>
              ))}
            </>
          )}

          <div className="btnrow" style={{ marginTop: 14 }}>
            <button className="btn primary" disabled={starting || running || missing.length > 0}
              onClick={() => void run()}>
              {starting ? <Spinner label="starting…" /> : running ? <Spinner label="running…" /> : "Run"}
            </button>
            {running && <button className="btn" onClick={() => void cancel()}>Cancel</button>}
          </div>
          {startErr && <div style={{ marginTop: 10 }}><ErrorBox title="could not start" detail={startErr} /></div>}
        </div>

        {/* ------------------------------------------------------ results (R) */}
        <div>
          {!job && (
            <div className="panel">
              <div className="empty">
                <div className="big">⛓️</div>
                fill the form and press <b>Run</b> — steps and results appear here as the
                pipeline produces them.
              </div>
            </div>
          )}

          {job && (
            <div className="panel">
              <div className="resulthead">
                <StatusChip status={job.status} />
                <span className="meta">{(job.duration_ms / 1000).toFixed(1)} s</span>
                <span className="meta mono">{job.id}</span>
              </div>

              <Steps steps={job.steps} />

              {Object.keys(job.info).length > 0 && (
                <div style={{ margin: "8px 0 10px" }}>
                  {Object.entries(job.info).map(([k, v]) => (
                    <div key={k} className="hint" style={{ fontFamily: "var(--mono)", fontSize: 12 }}>
                      <span className="k">{k}</span>: <RenderValue v={v} name={k} />
                    </div>
                  ))}
                </div>
              )}

              {job.error && (
                <div style={{ marginBottom: 10 }}>
                  <ErrorBox title={job.error.step ? `failed in “${job.error.step}”` : "failed"}
                    detail={{ message: job.error.message }} />
                  {job.error.trace && (
                    <details style={{ marginTop: 6 }}>
                      <summary style={{ cursor: "pointer", color: "var(--fg-dim)", fontSize: 12.5 }}>traceback</summary>
                      <pre style={{ overflow: "auto", fontSize: 11.5 }}>{job.error.trace}</pre>
                    </details>
                  )}
                </div>
              )}

              {def.input_mosaic !== false && jobImages.length > 0 && (
                <details className="viz">
                  <summary className="viz-caption" style={{ cursor: "pointer" }}>
                    inputs · {jobImages.length} image{jobImages.length > 1 ? "s" : ""} (click to show)
                  </summary>
                  <ImageGrid value={baseImages} />
                </details>
              )}

              {concrete.map((rd, i) => (
                <div className="viz" key={`${rd.field}-${rd.visualizer}-${i}`}>
                  {rd.field in fields ? (
                    <ResultBlock rd={rd} def={def} fields={fields} baseImages={baseImages}
                      history={[]} current={current} />
                  ) : (
                    <>
                      <div className="viz-caption">{rd.caption || rd.field}</div>
                      <div className="note">
                        {running ? <Spinner label="not produced yet…" /> : "not produced by this run"}
                      </div>
                    </>
                  )}
                </div>
              ))}

              {wildcard && Object.keys(fields).filter((f) => !covered.has(f)).map((f) => (
                <div className="viz" key={`wcard-${f}`}>
                  <div className="viz-caption">{f}</div>
                  <RenderValue v={fields[f]} name={f} />
                </div>
              ))}

              <details style={{ marginTop: 6 }}>
                <summary style={{ cursor: "pointer", color: "var(--fg-dim)", fontSize: 12.5 }}>
                  artifacts ({job.artifacts.length})
                </summary>
                <div style={{ marginTop: 6 }}>
                  {job.artifacts.map((a) => (
                    <span key={a.token} className="artifact">
                      <span className="kind">{a.content_type}</span>
                      <span>{bytesShort(a.size)}</span>
                      {a.extra && a.extra["shape"] !== undefined && <span className="kind">{String(a.extra["shape"])}</span>}
                      <a href={a.url} target="_blank" rel="noreferrer">open</a>
                    </span>
                  ))}
                </div>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Steps({ steps }: { steps: JobStep[] }) {
  if (!steps.length) return <div className="note"><Spinner label="queued…" /></div>;
  return (
    <div className="steps">
      {steps.map((s, i) => (
        <div key={i} className="step">
          <StatusChip status={s.status} />
          <span className="grow">
            {s.name}
            {s.message && <span className="hint" style={{ marginLeft: 6, color: "var(--err)" }}>{s.message}</span>}
            {s.total ? (
              <span className="bar" title={`${s.calls} / ${s.total} box calls`}>
                <span style={{ width: `${Math.min(100, (100 * s.calls) / s.total)}%` }} />
              </span>
            ) : null}
          </span>
          <span className="mono dim">
            {s.calls > 0 && `${s.calls}${s.total ? `/${s.total}` : ""} call${s.calls > 1 ? "s" : ""} · `}
            {fmtNum(Math.round(s.duration_ms / 100) / 10)} s
          </span>
        </div>
      ))}
    </div>
  );
}
