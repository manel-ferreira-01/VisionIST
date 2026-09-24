# Publishing `visionist-client` to PyPI

## Status: ✅ 0.1.1 published (2026-09-14)
https://pypi.org/project/visionist-client/0.1.1/ — installed from PyPI into a fresh
venv; import + both smoke suites pass. The name `visionist-client` is reserved.

## Original verification notes

- Name `visionist-client` is **free** on PyPI.
- `python -m build` produces clean sdist + pure-Python wheel (17 files,
  LICENSE bundled in the wheel as `dist-info/licenses/LICENSE`).
- Metadata checked: `License-Expression: GPL-3.0-only`,
  `Requires-Dist: protobuf>=7.35.1` (the vendored gencode enforces
  runtime >= gencode — do not lower), project URLs to the GitHub repo.
- Smoke tests pass in a clean venv **with and without torch**:
  - `tests/fake_box_smoke.py` — all cases (torch case skips gracefully)
  - `tests/codec_smoke.py` — all cases (torch cases skip gracefully)

## Steps (use your own account/token)

```bash
# 0. One-time: create a PyPI account at https://pypi.org/account/register/
#    (verify e-mail), then make a scoped API token:
#    https://pypi.org/manage/__user__/tokens/

# 1. Build
cd visionist_client
python -m build                     # -> dist/

# 2. Quick sanity (optional): install in a fresh venv and run the tests
python -m venv /tmp/relcheck && /tmp/relcheck/bin/pip install dist/*.whl
/tmp/relcheck/bin/python tests/fake_box_smoke.py
/tmp/relcheck/bin/python tests/codec_smoke.py

# 3. Upload (the FIRST upload reserves the name forever — deliberate)
pip install twine
python -m twine upload --non-interactive dist/*

# 4. Verify like a user would
pip download visionist-client==0.1.1 --no-deps -d /tmp/dl
pip install ./tmp/dl/visionist_client-0.1.1-py3-none-any.whl
python -c "from visionist_client import Visionist; print(Visionist)"
```

## Caveats

- **Uploads are irreversible** — versions can't be removed without a
  PyPI trust-and-abuse request. If anything looks off, stop before step 3.
- License is declared **GPL-3.0-only** (repo LICENSE, mirrored into this
  dir). Switch to a permissive license only after clearing the vendor
  pieces derived from upstream `jpcosteira/boxes` with that author.
- Re-releasing: bump `version` in `pyproject.toml`, rebuild, re-run the
  tests, upload. `0.1.1` below this tree works as the first public
  version (0.1.0 was never published); you may also rename to `0.1.0`
  before the first upload.
