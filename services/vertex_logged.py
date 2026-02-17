from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class VertexLogged:

    vertex: Any
    tracer: Any
    page: str
    dataset_id: Optional[str] = None

    def generate_json(
        self,
        phase: str,
        prompt: str,
        response_schema: Dict[str, Any],
        temperature: float,
        max_output_tokens: int,
        repair_attempts: int,
        token_expand_attempts: int,
        max_output_tokens_cap: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        err = None
        out: Optional[Dict[str, Any]] = None
        try:
            out = self.vertex.generate_json(
                prompt=prompt,
                response_schema=response_schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                repair_attempts=repair_attempts,
                token_expand_attempts=token_expand_attempts,
                max_output_tokens_cap=max_output_tokens_cap,
            )
            return out
        except Exception as e:
            err = str(e)
            raise
        finally:
            t1 = time.time()
            model = getattr(self.vertex, "model", None)

            safe_out: Dict[str, Any]
            if isinstance(out, dict):
                safe_out = out
            else:
                safe_out = {"output": out}

            self.tracer.generation(
                page=self.page,
                dataset_id=self.dataset_id,
                phase=phase,
                prompt=prompt,
                response_schema=response_schema,
                model=model or "",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                start_ts=t0,
                end_ts=t1,
                output=safe_out,
                error=err,
                metadata=metadata or {},
            )
