"""visionist-webui — a declarative web layer over a fleet of "boxes".

The core (``webui.core``) is deliberately *box-agnostic*: it knows the
envelope's shape and nothing about any specific box.  All box knowledge
lives in the YAML definitions under ``webui/boxes/`` — the same "smart
about shape, dumb about content" principle as ``visionist_client``.

    from webui.core import BoxDef, load_registry
"""

__version__ = "0.1.0"
