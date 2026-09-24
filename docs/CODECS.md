# CODECS — self-describing payload decoding  (design + rationale)

**Status: implemented** — `codec.py` (5 named codecs + `decode_with`),
`Result.encoding` threading, box declarations (lang_segm/tapnext/clip/sbert),
`zstandard` dependency, `tests/codec_smoke.py`, and the doc updates below.
This file remains the design record; acceptance criteria at the bottom.

Background: heavy payloads used to come back as raw `bytes` and the client
*guessed* how to decode them. That is now explicit and declared instead.

---

## 1. The problem (what we actually saw)

In the live fleet tour, `lang_segm` returned `results=bytes[2969]` — a
`zstd(compress) + pickle` blob (the documented cross-box contract: LangSAM
produces it, `folder_wd` consumes it). The client did **not** decode it, because
its decoder *guesses*:

```
decode_util.decode_payload():  JSON  →  torch  →  numpy  →  raw bytes
```

That guessing is the agnosticism leak. Two concrete failures:

1. **Mis-decode risk.** The numpy branch fires on *any* 4-byte-aligned blob and
   `np.frombuffer(buf, np.float32)` **succeeds and returns a garbage float
   array** (it only guards JPEG/PNG magic). A `zstd` blob is ~50% likely to be
   4-aligned, so `res.results` is sometimes a junk array. (This run escaped it
   only because 2969 % 4 == 1.)
2. **Opaque.** A caller can't tell what a `bytes` field means without knowing
   the box. That's per-box knowledge leaking into the caller, not the contract.

`tapnext`/`clip`/`sbert` (torch.save) are now decoded **by contract** — they declare their `encoding`. (`opencv` did the same after its migration to the standard envelope.) The guess chain survives only for boxes that never declare one.

## 2. The principle (already agreed, build on it)

> **The client should be smart about *shape* and dumb about *content*.**
> Content semantics (what a `bytes` blob *means*) belong to the **box's
> contract**, not the client. The core must keep naming **no box**; a codec is
> a generic `bytes -> object` function keyed by a **generic name**, and the box
> *selects* it by declaring it — the client never infers it from a box name.

Three rules:

- **Declare, don't guess.** The box says how its payload is encoded.
- **Named, generic codecs.** A small stable vocabulary of pure decoders. Adding
  a payload type = one new generic codec + one word in the contract. **Never** a
  per-box branch in the client.
- **Default = raw bytes.** A box that declares nothing (or an unknown codec)
  still returns usable raw `bytes`. That is the agnostic guarantee.

## 3. The contract (what a box adds to its response)

Add **one field** to the response `config_json`. It can be:

- a **string** codec name — applies to *every* `bytes`/`bytes-list` data field, or
- an **object** — a `{field_name: codec_name}` map for mixed fields.

```jsonc
// simple (whole response, one encoding)
{ "lang_sam": { "status": "done", "encoding": "zstd_pickle" } }

// precise (per field)
{ "tapnext":  { "status": "done",
    "encoding": { "tracks": "torch", "visibles": "torch" } } }
```

`encoding` is a **contract keyword** (generic, like `status`) — it is *not* a
box name, so the core may look for it without knowing the box. The client scans
the parsed config (top level, then each top-level section, shallowly) for an
`"encoding"` key and uses the first one found.

### Codec vocabulary (generic names, pure `bytes -> object`)

| name          | input                                | output                     | used by (today)          |
|---------------|--------------------------------------|----------------------------|--------------------------|
| `identity`    | raw bytes                            | `bytes` (unchanged)        | default / anything       |
| `json`        | UTF-8 JSON                           | `list`/`dict`              | (any JSON payload)       |
| `torch`       | `torch.save()` tensor / dict         | `Tensor` / `dict[Tensor]`  | tapnext, clip, sbert     |
| `numpy`       | `numpy.save` (`.npy`) blob, or raw float32 buffer (legacy) | `np.ndarray` | opencv (`np_to_bytes`)   |
| `zstd_pickle` | `zstd.compress(pickle.dumps(list))`  | decoded Python (usually `list`) | **lang_segm** (`results`) |

Each codec is a self-contained function with **no box name and no global state**.
`torch` and `zstd_pickle` depend on optional libs (`torch`, `zstandard`) — a
codec whose lib is missing must degrade to `identity` (raw bytes) + a warning,
**not** raise (the "client always returns something" rule).

## 4. Changes in `visionist_client` (the code to write)

> All in `src/visionist_client/`. Keep the core box-agnostic — none of this may
> branch on a box name.

### 4.1 New module `codec.py` (or add to `decode_util.py`)
```python
CODECS: dict[str, Callable[[bytes], Any]] = {
    "identity":    lambda b: b,
    "json":        _decode_json,          # existing _try_json body, minus the "is it json" sniff
    "torch":       _decode_torch,         # existing _try_torch body
    "numpy":       _decode_numpy,         # existing _try_numpy body (float32)
    "zstd_pickle": _decode_zstd_pickle,   # NEW: pickle.loads(zstandard.decompress(b))
}

def decode_with(payload: bytes, name: str):
    """Apply a NAMED codec. Unknown/missing lib -> identity (+warning). Never raises."""
```
Move the bodies of the existing private `_try_*` helpers into these (they already
exist in `decode_util.py`; just expose them by name). Add the new `_decode_zstd_pickle`
(`import zstandard`/`zstandard` lib; `pickle.loads(zstd.ZstdDecompressor().decompress(b))`).

### 4.2 `decode_util.decode_payload` — split into "declared" and "legacy auto"
Keep the old guess-chain **as a legacy fallback only** (so un-migrated boxes keep
working), but it is no longer the documented path. When a field has a declared
codec, call `decode_with(buf, codec)` directly and skip guessing.

### 4.3 `result.py` — thread the declared encoding through
`Result.from_envelope(envelope)`:
1. Parse `config` (already does).
2. New: `enc = _find_encoding(config)` → returns `None | str | dict`.
   `_find_encoding` scans `config` top-level then each one-level section for the
   `"encoding"` key (first hit wins).
3. When building `fields`, for each `b`/`bb` value: `codec = enc if isinstance(enc,str) else enc.get(field_name)`; if it's a codec name use `decode_with`, else fall back to the legacy `decode_payload`.
4. Expose `Result.encoding = enc` (the declared value) so callers can see *why*.

Back-compat: if `enc is None`, behaviour is **identical to today** → every
existing box keeps working unchanged.

### 4.4 `envelope` / `Visionist` — no change
The client sends what it's given; decoding is the only new logic. `Visionist.run`
signature unchanged.

### 4.5 `pyproject.toml`
Add `zstandard` to dependencies (it's already used by lang_segm/folder_wd).

## 5. Changes per box (one line each)

Add an `"encoding"` field to the box's **response** `config_json` (in the box
section). Minimal set — start with the two that are actually wrong today:

- **lang_segm** (`src/lang_sam_service.py`, success + error paths that set
  `data.results`): add `"encoding": "zstd_pickle"` to the `{"lang_sam": {...}}`
  status dict.
- **tapnext** (`src/tapnext_service.py`, `status:done`): add
  `"encoding": {"tracks":"torch","visibles":"torch","observation_matrix":"torch"}`.

Optional (currently decoded by the legacy chain) — for robustness, same one-liner:
- **clip**: `"encoding": {"image_emb":"torch","text_emb":"torch","similarity":"torch"}`.
- **sbert/textEmbedding**: `"encoding": {"embeddings":"torch","similarities":"torch"}`.

Already declared (reference):
- **opencv**: `"encoding": {"keypoints": "numpy", "descriptors": "numpy",
  "matches_inliers_a": "numpy", "matches_inliers_b": "numpy",
  "fundamental_matrix": "numpy"}` (np.save blobs).

> Boxes that add `encoding` only change their *self-description*; a new client
> decodes it, an old client ignores the extra key. Safe and reversible. Rebuild
> the affected images afterwards (they bake `src/` in at build time).

## 6. Tests to add (make it provable)

In `visionist_client/tests/fake_box_smoke.py` (or a new `tests/codec_smoke.py`):

1. A fake box whose response is `data={"results": zstd(pickle([...]))}` with
   `config={"codecbox":{"status":"done","encoding":"zstd_pickle"}}` → assert
   `res.results` is the **decoded list`, not `bytes`/`ndarray`.
2. A fake box declaring `encoding:"json"` with a JSON byte field → decodes.
3. A fake box declaring an *unknown* codec → returns **raw bytes** + `res.encoding`
   set (graceful, no raise).
4. A fake box with **no** `encoding` → byte field still decodes via legacy chain
   (back-compat unchanged).
5. `decode_with(b, "torch")` round-trips a `torch.save`/`torch.load` tensor.

Reuse the existing fake-servicer harness in `fake_box_smoke.py`; no GPU needed.

## 7. Docs to update

- `docs/gRPC_Services_Reference.md` → in the "Heavy results" section: replace the
  "client decodes for you" line with "declare `encoding` in the response config;
  codec table above; default raw."
- `visionist_client/README.md` → "Result object" / decode-order section: documented
  order is now *declared codec → (legacy auto)*; add the `codec.py` + examples.
- `images/lang_segm/README.md` → the client call can now rely on `res.results`
  being a decoded list (drop the manual `zstd`/`pickle` unwrap, keep as example
  for remote/old images).
- `docs/Architecture_Overview.md` conventions table → add row
  "payload encoding | declared in response `config_json` (`encoding`); codecs are generic; default raw".

## 8. Do / don't

**Do**
- Keep the core free of box names; `encoding` is the only new keyword.
- Default to `identity`; degrade (not raise) when a codec's lib is missing.
- Keep the legacy auto-chain as a *fallback* so un-migrated boxes still work.
- Add `zstandard` to the client deps.

**Don't**
- Don't branch on a box name anywhere in the client.
- Don't remove the legacy fallback yet (migrate boxes first, then optionally
  tighten `decode_payload` to `identity`-only once every box declares).
- Don't add conveniences-per-box in this task (that's a separate, optional layer).

## 9. Acceptance criteria (definition of done)

- [ ] `visionist_client` builds/imports; `codec.py` has the 5 named codecs + `decode_with`.
- [ ] `Result.from_envelope` reads `encoding` (string or per-field dict), exposes `res.encoding`.
- [ ] Fake-box test #1 (above) passes: lang_segm-style `results` returns a **decoded list**, `res.fields['results']` type is `list` (not `bytes`/`ndarray`).
- [ ] Test with unknown codec → raw bytes, no exception.
- [ ] Existing `fake_box_smoke.py` cases still pass (back-compat).
- [ ] `zstandard` in `pyproject.toml`.
- [ ] lang_segm (and ideally tapnext/clip/sbert) boxes emit `encoding` in their response config.
- [ ] Docs + `conventions` table updated.
- [ ] Re-run `fleet/hello.py`: the `lang_segm` row shows `results=list[...]` (decoded),
      not `results=bytes[2969]`.
- [ ] A new commit: `feat(visionist_client): codec registry + declared payload encoding`.

### Suggested order for the next session
1. `codec.py` + `decode_with` (pure, no I/O) + unit test for each codec.
2. `result.py` threading `encoding` through + res.encoding.
3. Fake-box codec tests → all green.
4. Add `"encoding"` to lang_segm (+ tapnext/clip/sbert) box sources.
5. `zstandard` dep; docs.
6. Rebuild the lang_segm image; `python fleet/hello.py` → confirm decoded `results`.
7. Commit.
