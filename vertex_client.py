# vertex_client.py
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import re

from google import genai


class VertexGenAIClient:
    """
    Клиент для Vertex Gemini, который:
    - аккуратно достаёт текст из resp.text или candidates[0].content.parts
    - пытается распарсить JSON напрямую или через вырезание блока
    - при подозрении на "обрезание" (max_output_tokens) увеличивает лимит и ретраит
    - если всё равно не получилось — делает repair-перегенерацию
    - возвращает понятные ошибки без "обрезанного JSON" в тексте
    """

    def __init__(
        self,
        project: str,
        location: str,
        model: str = "gemini-2.0-flash-001",
        debug: bool = False,
    ):
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model = model
        self._debug = debug

        # debug info (последний сырой ответ и причина завершения)
        self.last_raw: str = ""
        self.last_finish_reason: str = ""

    def generate_json(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        repair_attempts: int = 1,
        token_expand_attempts: int = 2,
        max_output_tokens_cap: int = 8192,
    ) -> Dict[str, Any]:
        cur_tokens = int(max_output_tokens)
        cur_temp = float(temperature)

        # 1) основной цикл с расширением токенов при обрезании
        last_raw = ""
        last_finish = ""

        for _ in range(max(0, token_expand_attempts) + 1):
            resp = self._call_vertex(
                prompt=prompt,
                response_schema=response_schema,
                temperature=cur_temp,
                max_output_tokens=cur_tokens,
            )

            raw = self._extract_text(resp)
            finish = self._get_finish_reason(resp)

            self.last_raw = raw
            self.last_finish_reason = finish

            last_raw = raw
            last_finish = finish

            parsed = self._try_parse_json(raw)
            if parsed is not None:
                return parsed

            if self._should_expand_tokens(raw, finish) and cur_tokens < max_output_tokens_cap:
                cur_tokens = min(max_output_tokens_cap, cur_tokens * 2)
                cur_temp = min(cur_temp, 0.2)
                continue

            break

        # 2) repair (по сути — перегенерация валидного JSON)
        bad_text = last_raw

        for _ in range(max(0, repair_attempts)):
            repair_prompt = self._build_repair_prompt(bad_text)

            cur_tokens = int(max_output_tokens)
            for _ in range(max(0, token_expand_attempts) + 1):
                resp2 = self._call_vertex(
                    prompt=repair_prompt,
                    response_schema=response_schema,
                    temperature=0.0,
                    max_output_tokens=cur_tokens,
                )

                raw2 = self._extract_text(resp2)
                finish2 = self._get_finish_reason(resp2)

                self.last_raw = raw2
                self.last_finish_reason = finish2

                parsed2 = self._try_parse_json(raw2)
                if parsed2 is not None:
                    return parsed2

                if self._should_expand_tokens(raw2, finish2) and cur_tokens < max_output_tokens_cap:
                    cur_tokens = min(max_output_tokens_cap, cur_tokens * 2)
                    continue

                bad_text = raw2
                break

        # 3) финальная ошибка — НОРМАЛЬНАЯ для пользователя
        raise RuntimeError(self._format_user_friendly_error(max_output_tokens))

    # ---------------- internal helpers ----------------

    def _call_vertex(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        temperature: float,
        max_output_tokens: int,
    ):
        try:
            return self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_output_tokens),
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            )
        except Exception as e:
            # это не токены, это реальный фейл вызова
            raise RuntimeError(f"Vertex call failed: {e}")

    def _extract_text(self, resp: Any) -> str:
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            return text

        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            raise RuntimeError("Vertex returned empty response (no candidates)")

        cand0 = candidates[0]
        content = getattr(cand0, "content", None)
        parts = getattr(content, "parts", None) or []

        chunks = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                chunks.append(t)

        text2 = "\n".join(chunks).strip()
        if not text2:
            raise RuntimeError("Vertex returned empty response (no text in parts)")
        return text2

    def _get_finish_reason(self, resp: Any) -> str:
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return ""
        cand0 = cands[0]
        fr = getattr(cand0, "finish_reason", None) or getattr(cand0, "finishReason", None) or ""
        return str(fr)

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            out = json.loads(text)
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            pass

        extracted = _extract_json_block(text)
        if extracted is None:
            return None

        try:
            out2 = json.loads(extracted)
            return out2 if isinstance(out2, dict) else None
        except json.JSONDecodeError:
            return None

    def _looks_truncated(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False

        # начинается как JSON, но не заканчивается закрывающей скобкой
        if s[0] in "{[" and s[-1] not in "}]":
            return True

        # оборванная строка
        if s.count('"') % 2 == 1:
            return True

        # не получилось вырезать сбалансированный блок
        if s[0] in "{[" and _extract_json_block(s) is None:
            return True

        return False

    def _should_expand_tokens(self, raw: str, finish_reason: str) -> bool:
        fr = (finish_reason or "").lower()

        # разные варианты у SDK/модели, делаем best-effort
        token_signals = [
            ("max", "token"),
            ("length", ""),
            ("output", "token"),
            ("token", "limit"),
        ]
        for a, b in token_signals:
            if a in fr and (b == "" or b in fr):
                return True

        return self._looks_truncated(raw)

    def _build_repair_prompt(self, bad_text: str) -> str:
        head = (bad_text or "")[:3000]
        return f"""
You returned invalid or incomplete JSON previously.

Return ONLY valid JSON (no markdown, no comments, no extra text).
The JSON MUST match the provided JSON schema exactly.
Make sure JSON is complete and not truncated.

Invalid output (truncated):
{head}
""".strip()

    def _format_user_friendly_error(self, max_output_tokens: int) -> str:
        finish = (self.last_finish_reason or "").strip()
        raw = (self.last_raw or "").strip()

        looks_truncated = self._looks_truncated(raw)
        finish_says_limit = self._should_expand_tokens(raw, finish)

        # 1) токены/обрезание
        if looks_truncated or finish_says_limit:
            msg = (
                "Генерация не удалась: ответ модели был обрезан из-за ограничения Max Tokens.\n\n"
                f"Текущее значение Max Tokens: {int(max_output_tokens)}.\n"
                "Что сделать:\n"
                "- увеличьте Max Tokens (например, в 2–4 раза)\n"
                "- или уменьшите Rows per table\n"
                "- или генерируйте таблицы по очереди/упростите схему (меньше колонок/ограничений)\n"
            )
            if finish:
                msg += f"\nТехническая причина завершения (finish_reason): {finish}"
            if self._debug and raw:
                msg += f"\n\nRaw head (debug):\n{raw[:800]}"
            return msg

        # 2) не токены: модель просто не соблюла JSON/schema
        msg = (
            "Генерация не удалась: модель не вернула валидный JSON, соответствующий schema.\n\n"
            "Что можно попробовать:\n"
            "- уменьшить Temperature (например, до 0.2)\n"
            "- уменьшить Rows per table\n"
            "- увеличить Max Tokens\n"
            "- упростить prompt (убрать лишние инструкции)\n"
        )
        if finish:
            msg += f"\nТехническая причина завершения (finish_reason): {finish}"
        if self._debug and raw:
            msg += f"\n\nRaw head (debug):\n{raw[:800]}"
        return msg


def _extract_json_block(s: str) -> Optional[str]:
    # убираем ```json ... ```
    s = re.sub(r"```json\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")

    obj_start = s.find("{")
    arr_start = s.find("[")

    if obj_start == -1 and arr_start == -1:
        return None

    start = obj_start if (obj_start != -1 and (arr_start == -1 or obj_start < arr_start)) else arr_start

    stack = []
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    continue
                stack.pop()
                if not stack:
                    return s[start : i + 1]

    return None