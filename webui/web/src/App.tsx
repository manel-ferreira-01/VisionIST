/** App shell: hash router (#/fleet, #/box/<defId>) + topbar nav.
 *  Nothing here knows a box's request shape — the defs drive the pages. */
import { Component, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, ApiError } from "./api";
import type { BoxDef } from "./api";
import { ErrorBox, Spinner } from "./ui";
import FleetPage from "./pages/FleetPage";
import ConsolePage from "./pages/ConsolePage";

function useHashRoute(): string {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const f = () => setHash(window.location.hash);
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return hash;
}

export function routeOf(hash: string): { page: "fleet" } | { page: "box"; id: string } {
  const path = hash.replace(/^#/, "") || "/";
  const m = path.match(/^\/box\/([^/]+)/);
  if (m) return { page: "box", id: decodeURIComponent(m[1]) };
  return { page: "fleet" };
}

/** Catch render-time crashes so a bad response can never unmount the whole
 *  app into a blank page: the user sees the error and can reload instead. */
class ErrorBoundary extends Component<
  { children: ReactNode },
  { err: Error | null; detail: string | null }
> {
  state: { err: Error | null; detail: string | null } = { err: null, detail: null };

  static getDerivedStateFromError(_err: Error): { err: Error } {
    return { err: _err };
  }

  componentDidCatch(_err: unknown, info: { componentStack?: string | null }): void {
    this.setState({ detail: (info?.componentStack || "").slice(0, 2000) });
  }

  render(): ReactNode {
    if (this.state.err) {
      return (
        <div className="panel">
          <h3>render crash — response detail hit a bug</h3>
          <p className="sub">
            The box response rendered, but the result panel threw {this.state.err.name}:{" "}
            {this.state.err.message}. The rest of the console is still alive;
            the offending detail is shown raw below.
          </p>
          <ErrorBox detail={{ error: String(this.state.err) }} />
          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: "pointer", color: "var(--fg-dim)", fontSize: 12.5 }}>component stack</summary>
            <pre style={{ overflow: "auto" }}>{this.state.detail || "(none)"}</pre>
          </details>
          <div className="btnrow" style={{ marginTop: 10 }}>
            <button className="btn" onClick={() => window.location.reload()}>reload webui</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const hash = useHashRoute();
  const route = routeOf(hash);
  const [defs, setDefs] = useState<BoxDef[] | null>(null);
  const [err, setErr] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    api.defs()
      .then((d) => setDefs(d.defs))
      .catch((e) => {
        setErr(e instanceof ApiError ? { status: e.status, ...e.detail } : { message: String(e) });
      });
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [hash]);

  let content: ReactNode;
  if (err) content = <ErrorBox title="could not load box definitions" detail={err} />;
  else if (!defs) content = <div className="empty"><Spinner label="loading definitions…" /></div>;
  else if (route.page === "box") content = <ConsolePage defs={defs} defId={route.id} />;
  else content = <FleetPage defs={defs} />;

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="#/fleet" style={{ color: "var(--fg)", textDecoration: "none" }}>
          📷 visionist <small>webui · contract-only fleet console</small>
        </a>
        {defs && (
          <nav>
            <a href="#/fleet" className={route.page === "fleet" ? "active" : ""}>fleet</a>
            {defs.map((d) => (
              <a
                key={d.id}
                href={`#/box/${d.id}`}
                className={route.page === "box" && route.id === d.id ? "active" : ""}
                title={d.name}
              >
                {d.id}
              </a>
            ))}
          </nav>
        )}
      </header>
      <main>
        <div className="page">
          <ErrorBoundary key={`${hash}-${String(err)}`}>{content}</ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
