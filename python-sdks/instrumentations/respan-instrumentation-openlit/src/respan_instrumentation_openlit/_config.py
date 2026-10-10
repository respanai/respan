"""Restore only OpenLIT configuration fields installed by this activation."""

from openlit._config import OpenlitConfig

_MISSING = object()
_FIELDS = (
    "_instance",
    "environment",
    "application_name",
    "pricing_info",
    "metrics_dict",
    "otlp_endpoint",
    "otlp_headers",
    "disable_batch",
    "capture_message_content",
    "disable_metrics",
    "disable_events",
    "capture_db_parameters",
    "max_content_length",
    "custom_span_attributes",
    "custom_metrics_attributes",
    "openlit_api_key",
    "openlit_url",
    "guard_pipeline",
)


def snapshot():
    return {key: vars(OpenlitConfig).get(key, _MISSING) for key in _FIELDS}


def restore(previous, installed):
    for key, old in previous.items():
        if vars(OpenlitConfig).get(key, _MISSING) is installed.get(key, _MISSING):
            if old is _MISSING:
                if key in vars(OpenlitConfig):
                    delattr(OpenlitConfig, key)
            else:
                setattr(OpenlitConfig, key, old)
