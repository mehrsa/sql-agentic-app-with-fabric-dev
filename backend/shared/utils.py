from datetime import datetime
import json
from types import SimpleNamespace
def to_dict_helper(instance):
    """Shared utility for converting model instances to dictionaries"""
    d = {}
    for column in instance.__table__.columns:
        value = getattr(instance, column.name)
        if isinstance(value, datetime):
            d[column.name] = value.isoformat()
        else:
            d[column.name] = value
    return d


def _to_json_primitive(value):
    """Recursively convert value to JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        # try to decode JSON strings into structures when possible
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return _to_json_primitive(parsed)
            except Exception:
                return value
        return value
    if isinstance(value, (list, tuple, set)):
        return [_to_json_primitive(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_primitive(v) for k, v in value.items()}
    # if object provides to_dict, use it
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _to_json_primitive(value.to_dict())
        except Exception:
            pass
    # try to extract public attributes
    try:
        attrs = {}
        for name in dir(value):
            if name.startswith("_"):
                continue
            attr = getattr(value, name)
            if callable(attr):
                continue
            # skip class/metadata descriptors
            if name in ("metadata", "registry", "query"):
                continue
            try:
                attrs[name] = _to_json_primitive(attr)
            except Exception:
                attrs[name] = str(attr)
        if attrs:
            return attrs
    except Exception:
        pass
    # fallback to string
    try:
        return str(value)
    except Exception:
        return None

def _serialize_message(msg):
    """Create a small JSON-safe dict for common LangGraph message objects."""
    # If it's already a dict, attempt to primitive-convert it
    if isinstance(msg, dict):
        return _to_json_primitive(msg)
    result = {"__class__": msg.__class__.__name__ if hasattr(msg, "__class__") else type(msg).__name__}
    # pick a set of attributes
    keys = ("content", "id", "additional_kwargs", "response_metadata", "tool_calls", "tool_call_id",
            "name", "usage_metadata", "tool_output", "role", "type", "status")
    for k in keys:
        try:
            v = getattr(msg, k, None)
        except Exception:
            v = None
        if v is not None:
            result[k] = _to_json_primitive(v)
    # also include a safe repr for debugging
    try:
        result["_repr"] = repr(msg)
    except Exception:
        result["_repr"] = str(msg)
    return result

def _serialize_messages(messages):
    msg_list = [_serialize_message(m) for m in (messages or [])]
    json_text = json.dumps(msg_list, indent=2)
    return json_text

def _to_obj(val):
    if isinstance(val, dict):
        return SimpleNamespace(**{k: _to_obj(v) for k, v in val.items()})
    if isinstance(val, list):
        return [_to_obj(x) for x in val]
    return val

def get_user_id():
    """Utility to get a fixed user ID for demo purposes."""
    user_id = 'user_5' #you can customize this as needed
    return user_id
