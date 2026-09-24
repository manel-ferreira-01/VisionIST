import os as _os
import sys as _sys

# The generated modules import `pipeline_pb2`, `aux` as top-level modules,
# so this directory has to be on sys.path before they are imported.
_HERE = _os.path.dirname(__file__)
if _HERE not in _sys.path:
    _sys.path.append(_HERE)
