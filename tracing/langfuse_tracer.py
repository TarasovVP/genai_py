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

    def ensure_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        lf = self._st.session_state.get("langfuse")
        if lf is None:
            return None

        if not self._st.session_state.get("trace_id"):
            self._st.session_state.trace_id = self.new_trace_id()

        try:
            return lf.trace(
                id=self._st.session_state.trace_id,
                name=name,
                user_id="streamlit_user",
                metadata=metadata or {},
            )
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
        page: str,
        dataset_id: Any,
        name: str,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        lf = self._st.session_state.get("langfuse")
        if lf is None:
            return

        tr = self.ensure_trace(name="data_assistant", metadata={"page": page, "dataset_id": dataset_id})
        if tr is None:
            return

        try:
            lf.event(
                trace_id=self._st.session_state.trace_id,
                name=name,
                level=level,
                message=message,
                metadata=metadata or {},
            )
        except Exception:
            return

    def span(
        self,
        page: str,
        dataset_id: Any,
        name: str,
        start_ts: float,
        end_ts: float,
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "ok",
    ) -> None:
        lf = self._st.session_state.get("langfuse")
        if lf is None:
            return

        tr = self.ensure_trace(name="data_assistant", metadata={"page": page, "dataset_id": dataset_id})
        if tr is None:
            return

        meta = dict(metadata or {})
        meta["status"] = status

        try:
            lf.span(
                trace_id=self._st.session_state.trace_id,
                name=name,
                start_time=datetime.utcfromtimestamp(start_ts).isoformat(timespec="seconds") + "Z",
                end_time=datetime.utcfromtimestamp(end_ts).isoformat(timespec="seconds") + "Z",
                metadata=meta,
            )
        except Exception:
            return

    def generation(
        self,
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
        lf = self._st.session_state.get("langfuse")
        if lf is None:
            return

        tr = self.ensure_trace(
            name="data_assistant",
            metadata={"page": page, "phase": phase, "dataset_id": dataset_id},
        )
        if tr is None:
            return

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
                trace_id=self._st.session_state.trace_id,
                name=phase,
                model=model,
                input=prompt,
                output=self.safe_preview(output, 3000) if output is not None else "",
                metadata=meta,
                start_time=datetime.utcfromtimestamp(start_ts).isoformat(timespec="seconds") + "Z",
                end_time=datetime.utcfromtimestamp(end_ts).isoformat(timespec="seconds") + "Z",
            )
        except Exception:
            return
