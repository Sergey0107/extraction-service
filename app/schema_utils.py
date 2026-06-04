"""Helpers for interpreting and normalizing JSON Schema extraction templates."""

import json
from typing import Any, Optional, Union


def build_default_json_schema(prompt: Optional[str]) -> dict[str, Any]:
    description = prompt or "Structured extraction result"
    return {
        "type": "object",
        "required": ["result"],
        "additionalProperties": False,
        "properties": {
            "result": {
                "type": "string",
                "description": description,
            }
        },
    }


def is_json_schema(schema: dict) -> bool:
    return schema.get("type") == "object" and isinstance(schema.get("properties"), dict)


def json_schema_to_template(schema: dict) -> dict:
    properties = schema.get("properties", {})
    return {
        key: json_schema_value_to_template(prop_schema)
        for key, prop_schema in properties.items()
    }


def json_schema_value_to_template(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return "string"

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next(
            (value for value in schema_type if value != "null"),
            schema_type[0] if schema_type else None,
        )

    if schema_type == "object":
        return json_schema_to_template(schema)
    if schema_type == "array":
        items = schema.get("items", {})
        return [json_schema_value_to_template(items)]
    if schema_type == "number":
        return "float"
    if schema_type == "integer":
        return "integer"
    if schema_type == "boolean":
        return "boolean"
    return "string"


def normalize_json_schema(
    schema: Optional[Union[dict, str]],
    prompt: Optional[str],
) -> dict[str, Any]:
    if isinstance(schema, dict) and is_json_schema(schema):
        return schema
    if isinstance(schema, str):
        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError:
            return build_default_json_schema(prompt)
        if isinstance(parsed, dict) and is_json_schema(parsed):
            return parsed
    return build_default_json_schema(prompt)
