import json
import logging

from src.web.logging import JsonFormatter


def test_json_formatter_emits_searchable_fields() -> None:
    formatter = JsonFormatter(service="fraud-api", environment="test")
    record = logging.LogRecord(
        name="fraud.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="prediction_completed",
        args=(),
        exc_info=None,
    )
    record.model = "lightgbm"
    record.batch_size = 32
    record.request_id = "request-123"

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "prediction_completed"
    assert payload["service"] == "fraud-api"
    assert payload["environment"] == "test"
    assert payload["request_id"] == "request-123"
    assert payload["model"] == "lightgbm"
    assert payload["batch_size"] == 32
    assert payload["timestamp"].endswith("Z")
