"""Loading of the vendored generated proto modules.

``visionist_client._pb.__init__`` puts its own directory on ``sys.path`` so the
generated modules (which import each other and ``aux`` as top-level) resolve.
Importing the package first guarantees that, then we grab the top-level modules.
"""

import importlib


def get():
    """Return ``(pipeline_pb2, pipeline_pb2_grpc, aux)``."""
    import visionist_client._pb  # noqa: F401  -> runs __init__, fixes sys.path

    pb2 = importlib.import_module("pipeline_pb2")
    pb2_grpc = importlib.import_module("pipeline_pb2_grpc")
    aux = importlib.import_module("aux")
    return pb2, pb2_grpc, aux
