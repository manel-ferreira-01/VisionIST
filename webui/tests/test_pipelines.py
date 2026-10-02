"""Pipelines end-to-end: HTTP -> JobRunner -> pipeline module -> several fake
boxes (real gRPC) -> emitted fields.  Uses the shipped ``sfm_video``
pipeline against fakes that speak the lightglue ``stream`` / moge ``infer``
/ sfm ``reconstruct`` contracts (shapes only — no real geometry)."""

import io
import json
import pathlib
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import FakeBox
from webui.app import create_app
from webui.config import AppEnv
from webui.core import BoxDef, CallRequest, build_call, ArtifactStore

WEBUI = pathlib.Path(__file__).resolve().parents[1]


def _npy(a) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(a))
    return buf.getvalue()


def _jpeg(n: int, w: int = 64, h: int = 48) -> bytes:
    img = Image.new("RGB", (w, h), (40 * n % 255, 90, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    env = AppEnv(
        host="127.0.0.1", port=0,
        data_dir=tmp_path / "data", boxes_dir=WEBUI / "boxes",
        artifact_ttl=600, max_artifact_bytes=50_000_000,
        max_upload_bytes=1_000_000, pipelines_dir=WEBUI / "pipelines",
    )
    with TestClient(create_app(env)) as c:
        yield c


def _box(std_pb, section: str, handler):
    pb2, pb2_grpc, aux = std_pb

    class Servicer(pb2_grpc.PipelineServiceServicer):
        def Process(self, request, context):
            cfg = json.loads(request.config_json) if request.config_json else {}
            sc = cfg.get(section, {})
            if sc.get("command") == "reset":
                return pb2.Envelope(config_json=json.dumps(
                    {section: {"status": "done", "action": "reset"}}))
            data = {k: aux.unwrap_value(v) for k, v in request.data.items()}
            status, arrays = handler(sc, data)
            enc = {k: "numpy" for k in arrays}
            env = pb2.Envelope(config_json=json.dumps({section: {**status, "encoding": enc}}))
            for k, a in arrays.items():
                env.data[k].CopyFrom(aux.wrap_value(_npy(a)))
            return env

    return FakeBox(Servicer(), pb2, pb2_grpc)


@pytest.fixture
def fake_lightglue(std_pb):
    """``stream``: N fixed keypoints drifting 1 px per frame, every reference
    in the window matches index i <-> i."""
    N = 40
    sessions: dict[str, int] = {}

    def handler(sc, data):
        p = sc.get("parameters", {})
        sid = p.get("session_id", "default")
        k = sessions.get(sid, 0)
        sessions[sid] = k + 1
        J = min(int(p.get("window", 3)), k)
        rows = [np.stack([10 + np.arange(N), 5 + 0.5 * np.arange(N) + (k - j)], -1)
                for j in range(J, -1, -1)]          # refs most-recent -> oldest, new last
        out = {"keypoints": np.stack(rows[::-1][1:] + rows[-1:]).astype(np.float32),
               "kp_counts": np.full(J + 1, N)}
        for j in range(1, J + 1):
            out[f"matches_{j}"] = np.stack([np.arange(N), np.arange(N)], -1)
        return {"status": "done", "window": J, "first_frame": J == 0}, out

    return _box(std_pb, "lightglue", handler)


def _fake_sfm(std_pb, fail: bool = False, delay: float = 0.0):
    def handler(sc, data):
        if delay:
            time.sleep(delay)
        if fail:
            return {"status": "error", "error": "too few points"}, {}
        tracks = np.load(io.BytesIO(data["tracks"]))
        depths = np.load(io.BytesIO(data["depths"]))
        F, P = tracks.shape[0] // 2, tracks.shape[1]
        assert depths.shape[0] == F
        cams = np.tile(np.eye(3, 4), (F, 1, 1))
        cams[:, 0, 3] = -0.1 * np.arange(F)
        status = {"status": "done", "solver": sc["parameters"]["solver"],
                  "num_frames": F, "num_points": P,
                  "reprojection_error": {"median": 1.5, "unit": "px"}}
        return status, {
            "cameras": cams, "points": np.random.default_rng(0).normal(size=(P, 3)) + [0, 0, 4],
            "frame_ids": np.arange(F), "point_ids": np.arange(P),
            "observed": np.isfinite(tracks[0::2]),
            "depth_scales": np.ones(F), "depth_offsets": np.zeros(F)}
    return _box(std_pb, "sfm", handler)


@pytest.fixture
def fake_sfm(std_pb):
    return _fake_sfm(std_pb)


def _seed(client, fake, name):
    r = client.post("/api/fleet", json={"name": name, "addr": fake.addr, "def_id": name})
    assert r.status_code == 201, r.text


def _upload(client, n):
    refs = []
    for i in range(n):
        r = client.post("/api/upload", files={"file": (f"f{i}.jpg", io.BytesIO(_jpeg(i)), "image/jpeg")})
        assert r.status_code == 201, r.text
        refs.append(r.json()["ref"])
    return refs


def _wait(client, job_id, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/pipelines/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {job['status']} after {timeout}s")


# --------------------------------------------------------------------------

def test_pipelines_listed_with_missing_boxes(client):
    body = client.get("/api/pipelines").json()
    p = {x["id"]: x for x in body["pipelines"]}["sfm_video"]
    assert p["uses"] == ["lightglue", "moge", "sfm"]
    assert p["missing"] == ["lightglue", "moge", "sfm"]        # empty fleet
    assert any(r["visualizer"] == "scene" for r in p["results"])
    assert "sfm_video" in client.get("/api/service").json()["pipelines"]


def test_run_refused_without_fleet_entries(client):
    r = client.post("/api/pipelines/sfm_video/run", json={"data": {"images": []}})
    assert r.status_code == 400
    assert r.json()["error"]["missing"] == ["lightglue", "moge", "sfm"]


def test_unknown_pipeline_and_job(client):
    assert client.post("/api/pipelines/nope/run", json={}).status_code == 404
    assert client.get("/api/pipelines/jobs/job_x").status_code == 404


def test_sfm_video_full_run(client, fake_lightglue, fake_moge, fake_sfm):
    for name, fake in (("lightglue", fake_lightglue), ("moge", fake_moge), ("sfm", fake_sfm)):
        _seed(client, fake, name)
    refs = _upload(client, 5)

    bad = client.post("/api/pipelines/sfm_video/run",
                      json={"data": {"images": refs}, "parameters": {"bogus": 1}})
    assert bad.status_code == 400 and "solver" in bad.json()["error"]["known"]

    r = client.post("/api/pipelines/sfm_video/run",
                    json={"data": {"images": refs}, "parameters": {"min_alive": 3}})
    assert r.status_code == 202, r.text
    job = _wait(client, r.json()["id"])
    assert job["status"] == "done", job["error"]

    steps = {s["name"]: s for s in job["steps"]}
    assert all(s["status"] == "done" for s in job["steps"])
    assert steps["lightglue · stream tracks"]["calls"] == 5 + 2      # reset + 5 frames + reset
    assert steps["moge · depth per frame"]["calls"] == 5
    assert steps["sfm · reconstruct"]["calls"] == 1
    assert job["info"]["tracks"] == 40 and job["info"]["solver"] == "linear"
    assert job["inputs"]["images"] == refs

    f = job["fields"]
    assert len(f["depths"]) == 5 and f["depths"][0]["depth"]["kind"] == "buffer"   # NaN corner
    assert f["track_depth"]["shape"] == [5, 40]
    assert f["sfm"]["status"] == "done"
    assert [row["frame"] for row in f["frames"]] == list(range(5))
    scene = f["scene"]
    assert scene["points"]["shape"] == [40, 3] and scene["colors"]["shape"] == [40, 3]
    assert scene["cameras"]["shape"] == [5, 3, 4]
    assert scene["image_size"] == [256, 320]                          # the moge map size
    dense = scene["dense_points"]
    assert dense["shape"][1] == 3 and 0 < dense["shape"][0] <= 300_000
    assert scene["dense_frames"]["shape"] == [dense["shape"][0]]
    # emitted buffers are fetchable
    tok = dense["url"].rsplit("/", 1)[-1]
    assert len(client.get(f"/api/file/{tok}").content) == dense["size"]


def test_box_error_fails_the_step(client, std_pb, fake_lightglue, fake_moge):
    fake = _fake_sfm(std_pb, fail=True)
    for name, b in (("lightglue", fake_lightglue), ("moge", fake_moge), ("sfm", fake)):
        _seed(client, b, name)
    r = client.post("/api/pipelines/sfm_video/run", json={"data": {"images": _upload(client, 3)}})
    job = _wait(client, r.json()["id"])
    assert job["status"] == "error"
    assert job["error"]["step"] == "sfm · reconstruct"
    assert "too few points" in job["error"]["message"]
    assert job["fields"]["sfm"]["status"] == "error"                  # emitted before the raise
    assert "depths" in job["fields"]                                  # earlier steps kept


def test_cancel(client, std_pb, fake_lightglue, fake_moge):
    fake = _fake_sfm(std_pb, delay=0.5)
    for name, b in (("lightglue", fake_lightglue), ("moge", fake_moge), ("sfm", fake)):
        _seed(client, b, name)
    r = client.post("/api/pipelines/sfm_video/run", json={"data": {"images": _upload(client, 3)}})
    job_id = r.json()["id"]
    t0 = time.time()
    while not any(s["name"].startswith("sfm") for s in client.get(f"/api/pipelines/jobs/{job_id}").json()["steps"]):
        assert time.time() - t0 < 20
        time.sleep(0.02)
    assert client.post(f"/api/pipelines/jobs/{job_id}/cancel").status_code == 200
    job = _wait(client, job_id)
    # the sfm call in flight finishes; the next boundary (the scene step) stops the run
    assert job["status"] == "cancelled"
    assert "scene" not in job["fields"]


def test_session_in_parameters():
    defn = BoxDef.model_validate({
        "id": "x", "name": "x", "box_key": "x", "command": {"values": ["stream"]},
        "session": {"key": "session_id", "placement": "parameters"},
        "parameters": [{"key": "window", "widget": "number", "default": 3}]})
    spec = build_call(defn, CallRequest(session_id="s1"), ArtifactStore())
    assert spec.config == {"x": {"command": "stream",
                                 "parameters": {"window": 3, "session_id": "s1"}}}
