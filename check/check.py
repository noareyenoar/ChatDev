"""Utilities for loading, validating design_0.4.0 workflows."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from runtime.bootstrap.schema import ensure_schema_registry_populated
from check.check_yaml import validate_design
from check.check_workflow import check_workflow_structure
from entity.config_loader import prepare_design_mapping
from entity.configs import DesignConfig, ConfigError
from schema_registry import iter_node_schemas
from utils.io_utils import read_yaml

class DesignError(RuntimeError):
    """Raised when a workflow design cannot be loaded or validated."""



def _allowed_node_types() -> set[str]:
    names = set(iter_node_schemas().keys())
    if not names:
        raise DesignError("No node types registered; cannot validate workflow")
    return names


def _ensure_supported(graph: Dict[str, Any]) -> None:
    """Ensure the MVP constraints are satisfied for the provided graph."""
    for node in graph.get("nodes", []) or []:
        nid = node.get("id")
        ntype = node.get("type")
        allowed = _allowed_node_types()
        if ntype not in allowed:
            raise DesignError(
                f"Unsupported node type '{ntype}' for node '{nid}'. Only {allowed} nodes are supported."
            )
        if ntype == "agent":
            agent_cfg = node.get("config") or {}
            if not isinstance(agent_cfg, dict):
                raise DesignError(f"Agent node '{nid}' config must be an object")
            for legacy_key in ["memory"]:
                if legacy_key in agent_cfg:
                    raise DesignError(
                        f"'{legacy_key}' is deprecated. Use the new graph-level memory stores for node '{nid}'."
                    )


def load_config(
    config_path: Path,
    *,
    fn_module: Optional[str] = None,
    set_defaults: bool = True,
    vars_override: Optional[Dict[str, Any]] = None,
) -> DesignConfig:
    """Load, validate, and sanity-check a workflow file."""

    ensure_schema_registry_populated()

    try:
        raw_data = read_yaml(config_path)
    except FileNotFoundError as exc:
        raise DesignError(f"Design file not found: {config_path}") from exc

    if not isinstance(raw_data, dict):
        raise DesignError("YAML root must be a mapping")

    if vars_override:
        merged_vars = dict(raw_data.get("vars") or {})
        merged_vars.update(vars_override)
        raw_data = dict(raw_data)
        raw_data["vars"] = merged_vars

    data = prepare_design_mapping(raw_data, source=str(config_path))

    schema_errors = validate_design(data, set_defaults=set_defaults, fn_module_ref=fn_module)
    if schema_errors:
        formatted = "\n".join(f"- {err}" for err in schema_errors)
        raise DesignError(f"Design validation failed for '{config_path}':\n{formatted}")

    try:
        design = DesignConfig.from_dict(data, path="root")
    except ConfigError as exc:
        raise DesignError(f"Design parsing failed for '{config_path}': {exc}") from exc

    logic_errors = check_workflow_structure(data)
    if logic_errors:
        formatted = "\n".join(f"- {err}" for err in logic_errors)
        raise DesignError(f"Workflow logical issues detected for '{config_path}':\n{formatted}")
    else:
        print("Workflow OK.")

    graph = data.get("graph") or {}
    _ensure_supported(graph)

    return design


def check_config(yaml_content: Any) -> str:
    ensure_schema_registry_populated()

    if not isinstance(yaml_content, dict):
        return "YAML root must be a mapping"

    # Skip placeholder resolution during save - users may configure env vars at runtime
    # Use yaml_content directly instead of prepare_design_mapping()
    schema_errors = validate_design(yaml_content)
    if schema_errors:
        formatted = "\n".join(f"- {err}" for err in schema_errors)
        return formatted

    logic_errors = check_workflow_structure(yaml_content)
    if logic_errors:
        formatted = "\n".join(f"- {err}" for err in logic_errors)
        return formatted

    graph = yaml_content.get("graph") or {}
    try:
        _ensure_supported(graph)
    except Exception as e:
        return str(e)

    return ""


def _parse_vars_override(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DesignError(f"Invalid --vars JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DesignError("--vars must decode to a JSON object")
    return parsed


def _parse_key_value_vars(raw_items: list[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in raw_items:
        if "=" not in item:
            raise DesignError(f"Invalid --var value '{item}'. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise DesignError(f"Invalid --var value '{item}'. KEY cannot be empty")
        parsed[key] = value
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and load a workflow YAML file")
    parser.add_argument("--path", required=True, help="Path to the workflow YAML file")
    parser.add_argument(
        "--fn-module",
        dest="fn_module",
        default=None,
        help="Optional module name or file path for edge helper functions",
    )
    parser.add_argument(
        "--vars",
        dest="vars_override",
        default=None,
        help="Optional JSON object overriding workflow vars during validation",
    )
    parser.add_argument(
        "--var",
        dest="var_items",
        action="append",
        default=[],
        help="Repeatable workflow variable override in KEY=VALUE format",
    )
    args = parser.parse_args()

    try:
        vars_override = _parse_vars_override(args.vars_override) or {}
        vars_override.update(_parse_key_value_vars(args.var_items))
        load_config(
            Path(args.path),
            fn_module=args.fn_module,
            vars_override=vars_override or None,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Validation OK: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())