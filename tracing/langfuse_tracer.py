from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4


class LangfuseTracer:
    def __init__(self, st: Any):
        self._st = st

    def new_trace_id(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:10]

    def _langfuse(self):
        return self._st.session_state.get("langfuse")

    def _iso_z(self, ts: float) -> str:
        return datetime.utcfromtimestamp(ts).isoformat(timespec="seconds") + "Z"

    def _ensure_trace_id(self) -> str:
        if not self._st.session_state.get("trace_id"):
            self._st.session_state.trace_id = self.new_trace_id()
        return self._st.session_state.trace_id

    def start_request_trace(
        self,
        *,
        page: str,
        dataset_id: Any,
        name: str = "data_assistant",
        user_id: str = "streamlit_user",
        metadata: Optional[Dict[str, Any]] = None,
        force_new_trace_id: bool = True,
    ) -> Optional[Any]:
        lf = self._langfuse()
        if lf is None:
            return None

        if force_new_trace_id:
            self._st.session_state.trace_id = self.new_trace_id()
            self._st.session_state.pop("_lf_trace_obj", None)

        trace_id = self._ensure_trace_id()

        meta = dict(metadata or {})
        meta.update({"page": page, "dataset_id": dataset_id})

        try:
            tr = lf.trace(
                id=trace_id,
                name=name,
                user_id=user_id,
                metadata=meta,
            )
            self._st.session_state["_lf_trace_obj"] = tr
            return tr
        except Exception:
            return None

    def ensure_trace(self, *, page: str, dataset_id: Any, name: str = "data_assistant") -> Optional[Any]:
        lf = self._langfuse()
        if lf is None:
            return None

        if self._st.session_state.get("_lf_trace_obj") is not None:
            return self._st.session_state.get("_lf_trace_obj")

        trace_id = self._ensure_trace_id()
        try:
            tr = lf.trace(
                id=trace_id,
                name=name,
                user_id="streamlit_user",
                metadata={"page": page, "dataset_id": dataset_id},
            )
            self._st.session_state["_lf_trace_obj"] = tr
            return tr
        except Exception:
            return None

    def safe_preview(self, obj: Any, limit: int = 2000) -> str:
        try:
            s = json.dumps(obj, ensure_ascii=False)
        except Exception:
            s = str(obj)
        if len(s) > limit:
            return s[:limit] + "…"
        return s

    def event(
        self,
        *,
        page: str,
        dataset_id: Any,
        name: str,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        lf = self._langfuse()
        if lf is None:
            return

        tr = self.ensure_trace(page=page, dataset_id=dataset_id, name="data_assistant")
        if tr is None:
            return

        trace_id = self._ensure_trace_id()

        try:
            lf.event(
                trace_id=trace_id,
                name=name,
                level=level,
                message=message,
                metadata=metadata or {},
            )
        except Exception:
            return

    def span(
        self,
        *,
        page: str,
        dataset_id: Any,
        name: str,
        start_ts: float,
        end_ts: float,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "ok",
    ) -> None:
        lf = self._langfuse()
        if lf is None:
            return

        tr = self.ensure_trace(page=page, dataset_id=dataset_id, name="data_assistant")
        if tr is None:
            return

        trace_id = self._ensure_trace_id()

        meta = dict(metadata or {})
        meta["status"] = status

        try:
            lf.span(
                trace_id=trace_id,
                name=name,
                start_time=self._iso_z(start_ts),
                end_time=self._iso_z(end_ts),
                metadata=meta,
            )
        except Exception:
            return

    def generation(
        self,
        *,
        page: str,
        dataset_id: Any,
        phase: str,
        prompt: str,
        response_schema: Optional[Dict[str, Any]],
        model: str,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        start_ts: float,
        end_ts: float,
        output: Any,
        error: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        lf = self._langfuse()
        if lf is None:
            return

        tr = self.ensure_trace(page=page, dataset_id=dataset_id, name="data_assistant")
        if tr is None:
            return

        trace_id = self._ensure_trace_id()

        meta = dict(metadata or {})
        meta.update(
            {
                "phase": phase,
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_schema_present": bool(response_schema),
                "output_preview": self.safe_preview(output, 1500) if output is not None else "",
                "error": error or "",
            }
        )

        try:
            lf.generation(
                trace_id=trace_id,
                name=phase,
                model=model,
                input=prompt,
                output=self.safe_preview(output, 3000) if output is not None else "",
                metadata=meta,
                start_time=self._iso_z(start_ts),
                end_time=self._iso_z(end_ts),
            )
        except Exception:
            return

    def guardrails_span(
        self,
        *,
        page: str,
        dataset_id: Any,
        start_ts: float,
        end_ts: float,
        allowed: bool,
        kind: str,
        reason: str,
        pii_counts: Optional[Dict[str, int]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.span(
            page=page,
            dataset_id=dataset_id,
            name="guardrails_check",
            start_ts=start_ts,
            end_ts=end_ts,
            status="ok" if allowed else "blocked",
            metadata={
                "allowed": bool(allowed),
                "kind": kind,
                "reason": reason,
                "pii_counts": pii_counts or {},
                **(meta or {}),
            },
        )

        if not allowed:
            self.event(
                page=page,
                dataset_id=dataset_id,
                name="guardrails_blocked",
                level="WARNING",
                message=reason or "Blocked by guardrails",
                metadata={
                    "kind": kind,
                    "pii_counts": pii_counts or {},
                    **(meta or {}),
                },
            )

    def chart_span(
        self,
        *,
        page: str,
        dataset_id: Any,
        start_ts: float,
        end_ts: float,
        chart_spec: Dict[str, Any],
        rows: int,
        status: str = "ok",
        error: str = "",
    ) -> None:
        self.span(
            page=page,
            dataset_id=dataset_id,
            name="chart_render",
            start_ts=start_ts,
            end_ts=end_ts,
            status=status,
            metadata={
                "chart_spec": self.safe_preview(chart_spec, 1000),
                "rows": int(rows),
                "error": error,
            },
        )
