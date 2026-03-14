import json

from overlay_constants import (
    CORNER_OPTIONS,
    DEFAULT_SETTINGS,
    DEFAULT_SETTINGS_PATH,
    MAX_OPACITY_PERCENT,
    MIN_OPACITY_PERCENT,
    MODEL_OPTIONS,
    PRIMARY_BOX_SIZE_MAX,
    PRIMARY_BOX_SIZE_MIN,
    USER_PREFERENCES_PATH,
)


def _clamp_int(value, low, high, fallback):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, parsed))


def _as_bool(value, fallback):
    if isinstance(value, bool):
        return value
    return fallback


def _sanitize_settings(raw):
    source = raw if isinstance(raw, dict) else {}
    return {
        "caption_box_size": _clamp_int(
            source.get("caption_box_size", source.get("font_size")),
            PRIMARY_BOX_SIZE_MIN,
            PRIMARY_BOX_SIZE_MAX,
            DEFAULT_SETTINGS["caption_box_size"],
        ),
        "opacity_percent": _clamp_int(source.get("opacity_percent"), MIN_OPACITY_PERCENT, MAX_OPACITY_PERCENT, DEFAULT_SETTINGS["opacity_percent"]),
        "show_raw_tokens": _as_bool(source.get("show_raw_tokens"), DEFAULT_SETTINGS["show_raw_tokens"]),
        "freeze_on_detection_loss": _as_bool(source.get("freeze_on_detection_loss"), DEFAULT_SETTINGS["freeze_on_detection_loss"]),
        "enable_llm_smoothing": _as_bool(source.get("enable_llm_smoothing"), DEFAULT_SETTINGS["enable_llm_smoothing"]),
        "model_selection": source.get("model_selection") if source.get("model_selection") in MODEL_OPTIONS else DEFAULT_SETTINGS["model_selection"],
        "show_latency": _as_bool(source.get("show_latency"), DEFAULT_SETTINGS["show_latency"]),
        "corner": source.get("corner") if source.get("corner") in CORNER_OPTIONS else DEFAULT_SETTINGS["corner"],
    }


def _read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_preferences_files():
    default_raw = _read_json(DEFAULT_SETTINGS_PATH)
    defaults = _sanitize_settings(default_raw if default_raw is not None else DEFAULT_SETTINGS)
    _write_json(DEFAULT_SETTINGS_PATH, defaults)

    user_raw = _read_json(USER_PREFERENCES_PATH)
    user = _sanitize_settings(user_raw if user_raw is not None else defaults)
    _write_json(USER_PREFERENCES_PATH, user)

    return defaults, user


def save_user_preferences(preferences):
    _write_json(USER_PREFERENCES_PATH, _sanitize_settings(preferences))
