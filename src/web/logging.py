import contextvars
import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.config import LoggingConfig

REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_standard_fields = set(logging.makeLogRecord({}).__dict__) | {
    "asctime",
    "message",
}


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp.replace("+00:00", "Z"),
            "level": record.levelname,
            "service": self._service,
            "environment": self._environment,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or _request_id.get()
        if request_id:
            payload["request_id"] = request_id

        for name, value in record.__dict__.items():
            if name not in _standard_fields and not name.startswith("_"):
                payload[name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


def configure_logging(config: LoggingConfig) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(config.service, config.environment))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(config.level)
    logging.captureWarnings(True)

    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._logger = logging.getLogger("fraud.http")

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = self._resolve_request_id(Headers(scope=scope))
        token = _request_id.set(request_id)
        started_at = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        fields = {
            "http_method": scope["method"],
            "http_path": scope["path"][:256],
        }
        try:
            await self._app(scope, receive, send_with_request_id)
        except Exception:
            self._logger.exception(
                "http_request_failed",
                extra={
                    **fields,
                    "http_status": status_code,
                    "duration_ms": self._duration_ms(started_at),
                },
            )
            raise
        else:
            self._logger.info(
                "http_request_completed",
                extra={
                    **fields,
                    "http_status": status_code,
                    "duration_ms": self._duration_ms(started_at),
                },
            )
        finally:
            _request_id.reset(token)

    @staticmethod
    def _resolve_request_id(headers: Headers) -> str:
        candidate = headers.get(REQUEST_ID_HEADER, "")
        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
        return uuid.uuid4().hex

    @staticmethod
    def _duration_ms(started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1_000, 3)
