"""Convert Pydantic models into JSON Schemas accepted by Claude's strict structured outputs.

Strict mode requires every object to declare ``additionalProperties: false`` and list
all of its properties as ``required``. ``$ref``s are inlined and validation keywords the
API does not enforce are dropped -- Pydantic re-validates (and clamps) the result anyway.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

_DROP_KEYS = {
    "default",
    "title",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "maxItems",
    "pattern",
    "format",
}


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    return _normalize(schema, defs)


def _normalize(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_normalize(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        resolved = copy.deepcopy(defs[name])
        extra = {k: v for k, v in node.items() if k != "$ref"}
        resolved.update(extra)
        return _normalize(resolved, defs)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROP_KEYS and not isinstance(value, dict):
            continue
        if key == "minItems" and isinstance(value, int) and value > 1:
            continue  # only 0/1 are supported
        if key == "properties":
            out[key] = {prop: _normalize(sub, defs) for prop, sub in value.items()}
        else:
            out[key] = _normalize(value, defs)

    if out.get("type") == "object" or "properties" in out:
        out["type"] = "object"
        out.setdefault("properties", {})
        out["required"] = list(out["properties"].keys())
        out["additionalProperties"] = False
    return out
