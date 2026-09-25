"""Tool-using agent loop, built on OpenAI's **Responses API**.

Why Responses API (not Chat Completions):
  - Responses API exposes paraphrased reasoning summaries via the
    `reasoning` output items (with `reasoning={"summary":"auto"}`),
    which lets us audit *why* the model made each tool call. Chat
    Completions only returns a `reasoning_tokens` count and hides
    the content.

Used by both `dataset_analyzer.py` and `target_prober.py`. Each agent
provides:
  - `system_prompt` (passed via the Responses API `instructions` field)
  - `user_prompt` (initial user-role input item)
  - `tools` (a list of Tool dataclass instances; tool names + JSON-Schema
     parameters + Python callables)
  - `max_turns`: hard cap.
  - `final_user_message`: optional structured final-report request,
     injected before the agent emits its final response (see below).

Per round we:
  1. Call `responses.create(...)` with the accumulated `input_items` list.
  2. Receive `resp.output` — a typed list of `reasoning`, `function_call`,
     and `message` items.
  3. If any `function_call` items: execute them in parallel
     (asyncio.gather), append `function_call_output` items, and continue
     the loop.
  4. Else (the agent has emitted only a `message`): treat as either
     - "I'm done with tools" → if `final_user_message` is supplied
       and hasn't been injected yet, inject it and force one more
       no-tools round to produce the structured final report.
     - Or "this is the final report" → return.

Multi-turn statelessness: we set `store=False` and pass back the prior
output items as input on every round. We add `include=["reasoning.encrypted_content"]`
so the API can preserve reasoning between rounds (per OpenAI cookbook).

The result we save (`AgentResult.output_items`) is the chronologically-
ordered list of every Responses item — system_prompt isn't there
(it's in the `instructions` param, saved separately) but every reasoning
summary, function_call, function_call_output, message, and injected
user message IS, so the web reader can render the full trajectory.

For backwards compatibility with the chat-completions-style web reader,
we also synthesize a `messages` list from the output_items.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
FORECASTING = HERE.parent.parent
sys.path.insert(0, str(FORECASTING))

from api_client import _get_openai_client, _get_forecasters  # noqa: E402


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON Schema dict
    fn: Callable[..., Any]     # actual python function

    def to_responses_api(self) -> dict:
        """Responses API tool shape — flat (no nested 'function' wrapper)."""
        return {
            "type":        "function",
            "name":        self.name,
            "description": self.description,
            "parameters":  self.parameters,
        }


@dataclass
class AgentResult:
    final_report: str
    n_turns: int
    trace: list[dict] = field(default_factory=list)
    error: str | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    final_user_message: str | None = None
    # NEW: full chronological list of Responses-API items (reasoning,
    # function_call, function_call_output, message, plus our synthesized
    # user_input items for the initial user prompt + final-message
    # injection). Each item is a plain dict.
    output_items: list[dict] = field(default_factory=list)
    # Backwards-compat: a chat-completions-style messages list synthesized
    # from output_items. The web reader's existing rendering uses this.
    messages: list[dict] = field(default_factory=list)


def _stringify_tool_result(result: Any, max_chars: int = 60_000) -> str:
    """Serialize a tool result to a string suitable for sending back to the
    agent. Caps total length at `max_chars` to keep context budgets sane."""
    if isinstance(result, str):
        s = result
    else:
        try:
            s = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            s = str(result)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n\n[...truncated; {len(s) - max_chars} chars omitted]"
    return s


def _strip_status(d: dict) -> dict:
    """Responses API output items include a `status` field (e.g. 'completed')
    that the API rejects on input. Strip it before passing back."""
    out = dict(d)
    out.pop("status", None)
    return out


def _items_to_messages(
    output_items: list[dict],
    system_prompt: str,
    initial_user_prompt: str,
) -> list[dict]:
    """Synthesize a chat-completions-style messages list from Responses API
    output_items. Used by the web reader for backwards-compat rendering.

    Roles emitted:
      - "system":            the system_prompt (passed via `instructions`)
      - "user":              initial user prompt + any injected user msgs
      - "reasoning_summary": custom pseudo-role for reasoning items
      - "assistant":         message items (with content text) AND function_call
                              items (with tool_calls field)
      - "tool":              function_call_output items
    """
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if initial_user_prompt:
        msgs.append({"role": "user", "content": initial_user_prompt})

    pending_assistant: dict | None = None  # accumulator: combine function_calls into one assistant message
    skipped_first_user_input = False  # skip the user_input item that mirrors initial_user_prompt

    def _flush_assistant():
        nonlocal pending_assistant
        if pending_assistant is not None:
            msgs.append(pending_assistant)
            pending_assistant = None

    for it in output_items:
        t = it.get("type")
        if t == "user_input":
            if not skipped_first_user_input and it.get("content") == initial_user_prompt:
                skipped_first_user_input = True
                continue  # already added as initial user prompt
            _flush_assistant()
            msgs.append({"role": "user", "content": it.get("content", "")})
        elif t == "reasoning":
            _flush_assistant()
            parts = it.get("summary") or []
            text = "\n\n".join((p.get("text") or "") for p in parts).strip()
            if text:
                msgs.append({"role": "reasoning_summary", "content": text,
                             "tool_call_id_pending": it.get("id", "")})
        elif t == "function_call":
            # Group consecutive function_call items into one assistant message
            tc = {
                "id":   it.get("call_id", ""),
                "type": "function",
                "function": {
                    "name":      it.get("name", ""),
                    "arguments": it.get("arguments", ""),
                },
            }
            if pending_assistant is None:
                pending_assistant = {"role": "assistant", "content": "", "tool_calls": [tc]}
            else:
                pending_assistant["tool_calls"].append(tc)
        elif t == "function_call_output":
            _flush_assistant()
            msgs.append({
                "role":         "tool",
                "tool_call_id": it.get("call_id", ""),
                "content":      it.get("output", ""),
            })
        elif t == "message":
            _flush_assistant()
            content_parts = it.get("content") or []
            text = "".join((c.get("text") or "") for c in content_parts)
            msgs.append({"role": "assistant", "content": text})
    _flush_assistant()
    return msgs


async def run_agent_loop(
    *,
    forecaster_alias: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[Tool],
    max_turns: int,
    final_user_message: str | None = None,
    plan_then_execute_message: str | None = None,
    timeout_s: float = 600.0,
    log_prefix: str | None = None,
    reasoning_effort_override: str | None = None,
) -> AgentResult:
    """Run the tool-using agent loop using OpenAI's Responses API.

    See module docstring for the full protocol. Returns AgentResult on
    success or with `error` set on failure. The agent is forced to emit
    a final report on its (max_turns + 1)th LLM call.

    `final_user_message`: structured-report request injected after the
    agent's first voluntary stop (or after max_turns is hit). The injection
    forces one additional no-tools round so the agent produces the
    structured final report.

    `log_prefix`: if provided, prints live progress per turn (reasoning
    summary preview, function-call name+args preview, tool result preview).

    `reasoning_effort_override`: override the forecaster's default
    reasoning_effort (e.g. "high" for orchestrator/analyzer/prober).
    """
    def _log(msg: str) -> None:
        if log_prefix is not None:
            print(f"        [{log_prefix}] {msg}", flush=True)

    cfg = _get_forecasters()[forecaster_alias]
    if cfg["provider"] != "openai":
        raise ValueError(f"agent_loop: only OpenAI provider supported, got {cfg['provider']!r}")
    client = _get_openai_client()
    model_id = cfg["model_id"]

    tools_by_name = {t.name: t for t in tools}
    api_tools = [t.to_responses_api() for t in tools]

    # input_items grows across rounds. Initialize with the user prompt.
    input_items: list[dict] = [{"role": "user", "content": user_prompt}]

    # output_items is the chronologically-ordered audit log we save.
    output_items: list[dict] = [
        # Synthesize an item for the initial user prompt so the web reader
        # can render it inline with the rest of the conversation.
        {"type": "user_input", "role": "user", "content": user_prompt},
    ]

    result = AgentResult(
        final_report="", n_turns=0,
        system_prompt=system_prompt, user_prompt=user_prompt,
        final_user_message=final_user_message,
    )
    trace: list[dict] = result.trace
    n_turns = 0

    final_msg_appended = False
    force_final_round = False
    # Plan-then-execute mode: round 0 is forced no-tools so the agent emits
    # a plan as a visible message (user_prompt should ask for the plan).
    # After receiving the plan, we inject `plan_then_execute_message` and
    # continue with normal tool-using rounds.
    plan_phase_pending = plan_then_execute_message is not None

    for turn in range(max_turns + 1):
        force_no_tools = (turn == max_turns) or force_final_round or plan_phase_pending

        # Inject final_user_message just before the (max_turns + 1)th call
        # if it hasn't been injected yet (e.g. agent is hitting the cap
        # without voluntarily stopping first).
        if (turn == max_turns or force_final_round) \
                and final_user_message and not final_msg_appended:
            input_items.append({"role": "user", "content": final_user_message})
            output_items.append({"type": "user_input", "role": "user",
                                  "content": final_user_message})
            final_msg_appended = True

        kwargs: dict[str, Any] = {
            "model":              model_id,
            "instructions":       system_prompt,
            "input":              input_items,
            "max_output_tokens":  cfg["max_completion_tokens"],
            "store":              False,
            "timeout":            timeout_s,
        }
        # Only reasoning models accept `reasoning` + `include=encrypted_content`.
        if "reasoning_effort" in cfg:
            kwargs["reasoning"] = {
                "effort":  reasoning_effort_override or cfg["reasoning_effort"],
                "summary": "auto",
            }
            kwargs["include"] = ["reasoning.encrypted_content"]
        if not force_no_tools and api_tools:
            kwargs["tools"] = api_tools

        t0 = time.time()
        try:
            resp = await client.responses.create(**kwargs)
        except Exception as e:
            result.n_turns = n_turns
            result.error = f"responses.create failed (turn={turn}): {e}"
            result.output_items = output_items
            result.messages = _items_to_messages(output_items, system_prompt, user_prompt)
            return result
        elapsed = time.time() - t0

        # Process every output item: log, append to output_items, append
        # a stripped-status copy to input_items for the next round.
        function_calls: list[Any] = []
        message_text = ""
        n_reasoning_items = 0
        for item in resp.output:
            d = item.model_dump()
            output_items.append(d)
            input_items.append(_strip_status(d))
            if item.type == "reasoning":
                n_reasoning_items += 1
            elif item.type == "function_call":
                function_calls.append(item)
            elif item.type == "message":
                for c in item.content or []:
                    if hasattr(c, "text") and c.text:
                        message_text += c.text

        # Trace entry for this LLM round.
        trace.append({
            "turn":           turn,
            "elapsed_s":      round(elapsed, 2),
            "n_reasoning":    n_reasoning_items,
            "n_function_calls": len(function_calls),
            "message_chars":  len(message_text),
            "force_no_tools": force_no_tools,
        })

        # Live log: turn header + reasoning summary previews + function calls
        _log(f"turn {turn} ({elapsed:.1f}s): "
             f"{n_reasoning_items} reasoning, {len(function_calls)} fn_call(s)"
             + (f", message={message_text[:120]!r}" if message_text else ""))
        for item in resp.output:
            if item.type == "reasoning":
                for s in (item.summary or []):
                    text = (s.text or "").replace("\n", " ").strip()
                    if text:
                        _log(f"  [t{turn}] reasoning: {text[:200]!r}")
            elif item.type == "function_call":
                args = item.arguments
                if len(args) > 200:
                    args = args[:200] + "…"
                _log(f"  [t{turn}] fn_call: {item.name}({args})")

        # No function calls → either: (a) plan phase complete (plan
        # received as a message), inject "now execute" and re-enable tools;
        # (b) voluntary stop, inject final_user_message + force one more
        # no-tools round; (c) actual final report.
        if not function_calls:
            if plan_phase_pending:
                _log(f"  (plan received, {len(message_text)} chars) → "
                     f"injecting plan_then_execute_message")
                input_items.append({"role": "user", "content": plan_then_execute_message})
                output_items.append({"type": "user_input", "role": "user",
                                      "content": plan_then_execute_message})
                plan_phase_pending = False
                continue
            if final_user_message and not final_msg_appended:
                _log(f"  (interim stop, {len(message_text)} chars) → "
                     f"injecting final_user_message and forcing one more round")
                input_items.append({"role": "user", "content": final_user_message})
                output_items.append({"type": "user_input", "role": "user",
                                      "content": final_user_message})
                final_msg_appended = True
                force_final_round = True
                continue
            _log(f"  ✓ final report ({len(message_text)} chars): "
                 f"{message_text[:160]!r}…")
            result.final_report = message_text
            result.n_turns = n_turns
            result.output_items = output_items
            result.messages = _items_to_messages(output_items, system_prompt, user_prompt)
            return result

        # Execute function calls in parallel.
        n_turns += len(function_calls)

        async def _exec_one(fc):
            tool = tools_by_name.get(fc.name)
            if tool is None:
                return f"ERROR: unknown tool {fc.name!r}"
            try:
                args = json.loads(fc.arguments)
                tool_out = tool.fn(**args)
                if inspect.isawaitable(tool_out):
                    tool_out = await tool_out
                return _stringify_tool_result(tool_out)
            except Exception as e:
                return f"ERROR: tool {fc.name} raised: {e}"

        results = await asyncio.gather(*[_exec_one(fc) for fc in function_calls])
        for fc, res_str in zip(function_calls, results):
            fco = {
                "type":    "function_call_output",
                "call_id": fc.call_id,
                "output":  res_str,
            }
            input_items.append(fco)
            output_items.append(fco)
            _log(f"  [t{turn}→t{turn+1}] tool_result ({fc.name}, "
                 f"{len(res_str)} chars): {res_str[:200].strip()!r}…")
            trace[-1].setdefault("tool_results", []).append({
                "name":           fc.name,
                "result_chars":   len(res_str),
                "result_preview": res_str[:200],
            })

    # Out of turns and no final report — shouldn't happen because turn ==
    # max_turns forces no_tools.
    result.n_turns = n_turns
    result.error = "agent loop exceeded max_turns + 1 without emitting a final response"
    result.output_items = output_items
    result.messages = _items_to_messages(output_items, system_prompt, user_prompt)
    return result
