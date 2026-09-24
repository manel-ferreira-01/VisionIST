"""build_call(): wire request -> Envelope-ready shape.  Pure — no network.

Covers the payload layout rules (namespaced vs flat), command/parameter
defaults, session-id injection, reset semantics, token resolution, and the
strictness that catches user typos.
"""

import pathlib

import pytest

from webui.core import (
    ArtifactStore, BoxDef, CallBuildError, CallRequest,
    build_call, load_registry,
)

BOXES_DIR = pathlib.Path(__file__).resolve().parents[1] / "boxes"


@pytest.fixture
def reg():
    return load_registry(BOXES_DIR)


@pytest.fixture
def store():
    return ArtifactStore(ttl=60)


def _put(store, b: bytes) -> str:
    return store.add(b, "image/jpeg")


# ------------------------------------------------------------------- layout

def test_lang_sam_namespaced_layout(reg, store):
    spec = build_call(reg.get("lang_sam"), CallRequest(
        data={"images": [b"img1", b"img2"]},
        section={"text_prompt": ["a dog", "the road"]},
        parameters={"box_threshold": 0.4},
    ), store)
    assert spec.config == {
        "lang_sam": {
            "command": "segment",
            "parameters": {"box_threshold": 0.4, "text_threshold": 0.25,
                           "device": "auto"},
            "text_prompt": ["a dog", "the road"],
        }
    }
    assert spec.data["images"] == [b"img1", b"img2"]
    assert spec.method == "Process"
    assert spec.reset_first is True        # stateless -> safe no-op reset
    assert spec.session_id is None


def test_clip_defaults_and_reset(reg, store):
    d = reg.get("clip")
    spec = build_call(d, CallRequest(data={"texts": ["a dog"]}), store)
    assert spec.config["clip"]["command"] == "encode"
    assert spec.config["clip"]["parameters"]["model"] == "ViT-B/32"
    assert spec.reset_first is True

    spec = build_call(d, CallRequest(command="reset"), store)
    assert spec.config["clip"]["command"] == "reset"
    assert spec.reset_first is False        # the call *is* the reset


def test_vggt_namespaced_layout(reg, store):
    d = reg.get("vggt")
    spec = build_call(d, CallRequest(
        data={"images": [b"f1", b"f2", b"f3"]},
        parameters={"conf_threshold": 25},
    ), store)
    # namespaced under the box key; device only present when the user picks one
    assert set(spec.config) == {"vggt"}
    assert spec.config["vggt"]["command"] == "reconstruct"
    assert spec.config["vggt"]["parameters"] == {"conf_threshold": 25}
    assert spec.reset_first is True          # stateless box -> safe no-op reset

    spec = build_call(d, CallRequest(command="reset"), store)
    assert spec.config["vggt"]["command"] == "reset"
    assert spec.reset_first is False         # the call *is* the reset

    spec = build_call(d, CallRequest(
        data={"images": [b"f1"]}, parameters={"device": "cpu"}), store)
    assert spec.config["vggt"]["parameters"]["device"] == "cpu"


def test_tapnext_session_injection(reg, store):
    d = reg.get("tapnext")
    spec = build_call(d, CallRequest(
        data={"images": b"frame.jpg"},
        session_id="alice-2025",
    ), store)
    assert spec.config["tapnext"]["session_id"] == "alice-2025"
    assert spec.config["tapnext"]["command"] == "track"
    assert spec.reset_first is False       # never break a live session
    assert spec.session_id == "alice-2025"
    assert spec.data["images"] == b"frame.jpg"


def test_out_of_scope_boxes_absent(reg, store):
    """opencv_box is skipped until it migrates to the shared envelope
    (see README) — the registry is the source of truth."""
    ids = {d.id for d in reg}
    assert "opencv" not in ids


# ------------------------------------------------------------------- strict

def test_unknown_command_rejected(reg, store):
    with pytest.raises(CallBuildError) as ei:
        build_call(reg.get("lang_sam"), CallRequest(
            data={"images": [b"x"]}, command="fly"), store)
    assert "segment" in ei.value.detail["known"]


def test_unknown_parameter_rejected(reg, store):
    d = reg.get("tapnext")
    with pytest.raises(CallBuildError) as ei:
        build_call(d, CallRequest(data={"images": b"x"},
                                 parameters={"bogus": 1}), store)
    assert "bogus" not in ei.value.detail["known"]


def test_unknown_data_field_rejected(reg, store):
    with pytest.raises(CallBuildError) as ei:
        build_call(reg.get("clip"), CallRequest(data={"pics": [b"x"]}), store)
    assert "images" in ei.value.detail["known"]


def test_unknown_section_key_rejected(reg, store):
    with pytest.raises(CallBuildError) as ei:
        build_call(reg.get("lang_sam"), CallRequest(
            data={"images": [b"x"]}, section={"prompt": ["a"]}), store)
    assert "text_prompt" in ei.value.detail["known"]


def test_session_on_stateless_box_rejected(reg, store):
    with pytest.raises(CallBuildError, match="no session"):
        build_call(reg.get("clip"), CallRequest(
            data={"texts": ["a"]}, session_id="s1"), store)


def test_missing_required_data_rejected(reg, store):
    with pytest.raises(CallBuildError, match="images"):
        build_call(reg.get("lang_sam"), CallRequest(
            section={"text_prompt": ["a"]}), store)


def test_missing_required_section_rejected(reg, store):
    with pytest.raises(CallBuildError, match="text_prompt"):
        build_call(reg.get("lang_sam"), CallRequest(data={"images": [b"x"]}), store)


def test_unknown_action_rejected(reg, store):
    from webui.core.schema import ActionDef
    d = BoxDef(id="multi", name="multi", box_key="multi",
               actions=[ActionDef(name="alpha"), ActionDef(name="beta")])
    with pytest.raises(CallBuildError) as ei:
        build_call(d, CallRequest(action="gamma"), store)
    assert "alpha" in ei.value.detail["known"]


def test_method_without_contract_rejected():
    from webui.core.schema import ActionDef
    d = BoxDef(id="pre", name="pre", box_key="pre",
               actions=[ActionDef(name="detect", method="DetectSequence")])
    with pytest.raises(CallBuildError, match="not the shared contract RPC"):
        build_call(d, CallRequest(action="detect"), ArtifactStore())


# -------------------------------------------------------------------- tokens

def test_upload_tokens_resolve_to_bytes(reg, store):
    tok = _put(store, b"jpeg-bytes")
    spec = build_call(reg.get("lang_sam"), CallRequest(
        data={"images": ["@" + tok]},
        section={"text_prompt": ["a"]}), store)
    assert spec.data["images"] == [b"jpeg-bytes"]


def test_unknown_token_rejected(reg, store):
    with pytest.raises(Exception, match="token"):
        build_call(reg.get("lang_sam"), CallRequest(
            data={"images": ["@no_such_token"]},
            section={"text_prompt": ["a"]}), store)


def test_single_value_field_unwraps_one_item(reg, store):
    from webui.core.schema import InputField, ResultDef
    d = BoxDef(id="single", name="single", box_key="single",
               inputs=[InputField(field="frame", widget="image_upload", kind="b",
                                  multiple=False, required=False)],
               results=[ResultDef(field="*", visualizer="json")])
    spec = build_call(d, CallRequest(data={"frame": [b"only"]}), store)
    assert spec.data["frame"] == b"only"
    with pytest.raises(CallBuildError, match="single value"):
        build_call(d, CallRequest(data={"frame": [b"a", b"b"]}), store)


def test_tapnext_images_is_a_list(reg, store):
    """Live-verified: the real tapnext box requires data.images to be a LIST
    (even for single-frame tracking)."""
    d = reg.get("tapnext")
    f = d.input_fields()["images"]
    assert f.kind == "bb" and f.multiple
    spec = build_call(d, CallRequest(data={"images": [b"frame"]}), store)
    assert spec.data["images"] == [b"frame"]


def test_int_coercion_left_to_client(reg, store):
    """ints in parameters are fine — visionist_client coerces when enveloping."""
    d = reg.get("tapnext")
    spec = build_call(d, CallRequest(
        data={"images": b"frame"}, parameters={"grid_size": 8}), store)
    assert spec.config["tapnext"]["parameters"]["grid_size"] == 8


def test_spec_metadata_is_jsonable():
    """config + metadata cross the JSON boundary; data may carry binary blobs
    (that's what gets shipped to the box)."""
    import json
    from webui.core import CallSpec
    spec = CallSpec(data={"images": [b"x"]}, config={"a": {"b": [1, 2]}},
                    method="Process", action="match", session_id="s")
    payload = {"config": spec.config, "method": spec.method,
               "action": spec.action, "session_id": spec.session_id}
    assert json.dumps(payload)
    assert spec.reset_first in (True, False)
