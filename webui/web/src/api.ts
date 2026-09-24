/** Shared API + types for the visionist-webui backend.
 *  Wire contract documented in webui/README.md (API + wire rules). */

// ---------------------------------------------------------------- definitions

export interface InputField {
  field: string;
  widget: string;
  kind: string;
  multiple: boolean;
  required: boolean;
  default?: unknown;
  helper?: string | null;
  constraint?: string | null;
  placeholder?: string | null;
}

export interface ParamDef {
  key: string;
  widget: string;
  default?: unknown;
  required?: boolean;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  values?: string[] | null;
  placeholder?: string | null;
}

export interface ActionDef {
  name: string;
  method: string;
  parameters: ParamDef[];
  note?: string | null;
}

export interface CommandSpec {
  values: string[];
  default?: string | null;
}

export interface SessionDef {
  key: string;
  auto_generate: boolean;
  actions: string[];
  note?: string | null;
}

export interface LayerDef {
  prop: string;
  layer: string;
  color_by_index?: boolean;
  opacity?: number | null;
}

export interface ResultDef {
  field: string;
  visualizer: string;
  caption?: string | null;
  base?: string | null;
  only_if_missing?: string | null;
  layers: LayerDef[];
  inputs: Record<string, string>;
  note?: string | null;
  params: Record<string, unknown>;
}

export interface BoxDef {
  id: string;
  name: string;
  box_key: string | null;
  flat_config: boolean;
  method: string;
  docs?: string | null;
  note?: string | null;
  experimental: boolean;
  input_mosaic?: boolean;
  inputs: InputField[];
  actions: ActionDef[];
  command: CommandSpec | null;
  parameters: ParamDef[];
  section: ParamDef[];
  session: SessionDef | null;
  results: ResultDef[];
}

export interface DefsPayload {
  defs: BoxDef[];
  vocabulary: {
    widgets: string[];
    visualizers: string[];
    overlay_layers: string[];
    value_kinds: string[];
  };
}

// ---------------------------------------------------------------------- fleet

export interface Probe {
  at: number;
  addr: string;
  reachable?: boolean;
  reflection?: boolean;
  methods?: string[];
  service?: string;
  error?: string;
}

export interface FleetEntry {
  id: string;
  name: string;
  addr: string;
  def_id: string | null;
  note?: string | null;
  added: number;
  last_probe: Probe | null;
}

// ---------------------------------------------------------------------- calls

/** Serialized field values (see webui/core/serialize.py). */
export type JSONValue = null | boolean | number | string | JSONValue[] | { [k: string]: JSONValue };

export interface SerArray { kind: "array"; dtype: string; shape: number[]; values: JSONValue; }
export interface SerBuffer { kind: "buffer"; url: string; dtype: string; shape: number[]; size: number; }
export interface SerFile { kind: "file"; url: string; mime: string; size: number; extra?: Record<string, unknown>; }

/** A serialized *reference* value (artifact) — discriminable by ``kind``. */
export type SerRef = SerArray | SerBuffer | SerFile;

export type SerValue = JSONValue | SerRef;

export interface Artifact {
  token: string;
  ref: string;
  url: string;
  content_type: string;
  size: number;
  extra: Record<string, unknown>;
}

export interface CallResult {
  fields: Record<string, unknown>;
  status?: string | null;
  error?: string | null;
  runtime?: number | null;
  config_extra: Record<string, unknown>;
  declared_encoding: string | Record<string, string> | null;
  box: string;
  addr: string;
  method: string;
  action: string | null;
  session_id: string | null;
  duration_ms: number;
  artifacts: Artifact[];
}

export interface UploadResult {
  token: string;
  ref: string;
  size: number;
  content_type: string;
}

// ----------------------------------------------------------------------- api

export class ApiError extends Error {
  status: number;
  detail: Record<string, unknown>;
  constructor(status: number, detail: Record<string, unknown>) {
    super(String(detail["message"] || JSON.stringify(detail)));
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  // build a proper headers object (a spread with `headers: undefined` would
  // wipe the content-type and some proxies then mangle the body)
  const headers: Record<string, string> = {};
  if (init?.body && !(init.body instanceof FormData)) {
    headers["content-type"] = "application/json";
  }
  Object.assign(headers, init?.headers || {});
  const r = await fetch(path, { ...init, headers });
  if (!r.ok) {
    let detail: Record<string, unknown> = { message: r.statusText };
    try {
      const j = (await r.json()) as { error?: Record<string, unknown> };
      if (j.error) detail = j.error;
    } catch {
      /* not json */
    }
    throw new ApiError(r.status, detail);
  }
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

export const api = {
  root: () => req<{ service: string; version: string; defs: string[] }>("/"),
  defs: () => req<DefsPayload>("/api/defs"),
  def: (id: string) => req<BoxDef>(`/api/defs/${encodeURIComponent(id)}`),
  fleet: () => req<{ entries: FleetEntry[] }>("/api/fleet"),
  fleetAdd: (b: { name: string; addr: string; def_id?: string | null; note?: string | null }) =>
    req<FleetEntry>("/api/fleet", { method: "POST", body: JSON.stringify(b) }),
  fleetPatch: (id: string, b: Partial<{ name: string; addr: string; def_id: string | null; note: string | null }>) =>
    req<FleetEntry>(`/api/fleet/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(b) }),
  fleetDel: (id: string) => req<void>(`/api/fleet/${encodeURIComponent(id)}`, { method: "DELETE" }),
  fleetProbe: (id: string, timeout = 4) =>
    req<FleetEntry>(`/api/fleet/${encodeURIComponent(id)}/probe?timeout=${timeout}`, { method: "POST" }),
  upload: (file: File) => uploadViaForm(file),
  file: (token: string) => `/api/file/${encodeURIComponent(token)}`,
  call: (b: {
    fleet_id: string;
    def_id?: string | null;
    data?: Record<string, unknown>;
    parameters?: Record<string, unknown>;
    section?: Record<string, unknown>;
    command?: string | null;
    action?: string | null;
    session_id?: string | null;
    timeout?: number;
  }) => req<CallResult>("/api/call", { method: "POST", body: JSON.stringify(b) }),
};

function fd(file: File): FormData {
  const f = new FormData();
  f.append("file", file);
  return f;
}

/** Upload via raw fetch (FormData sets its own content-type with boundary). */
async function uploadViaForm(file: File): Promise<UploadResult> {
  const body = fd(file);
  const r = await fetch("/api/upload", { method: "POST", body });
  if (!r.ok) {
    let detail: Record<string, unknown> = { message: r.statusText };
    try {
      detail = ((await r.json()) as { error?: Record<string, unknown> }).error || detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(r.status, detail);
  }
  return (await r.json()) as UploadResult;
}
