"""OpenAI-compatible proxy server for Tinker LoRA models.

Exposes a /v1/chat/completions endpoint so that LiteLLM/BLOOM can call
Tinker fine-tuned models as if they were an OpenAI API.

Usage:
    python tinker_proxy.py \
        --model-path "tinker://UUID:train:0/sampler_weights/final" \
        --tokenizer "Qwen/Qwen3-32B" \
        --model-name "qwen3-32b-cu-ft" \
        --port 8031

Then run BLOOM with:
    OPENAI_API_BASE=http://localhost:8031/v1 python run_bloom.py --model qwen3-32b-cu-ft
"""

import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker.types import SamplingParams
from transformers import AutoTokenizer


class TinkerProxy:
    def __init__(self, model_path: str | None, tokenizer_id: str, model_name: str,
                 base_model: str | None = None):
        if not model_path and not base_model:
            raise ValueError("Either --model-path or --base-model must be provided")
        self.model_name = model_name
        print(f"Loading tokenizer: {tokenizer_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if model_path:
            print(f"Connecting to Tinker FT weights: {model_path}")
            self.service_client = tinker.ServiceClient()
            self.sampling_client = self.service_client.create_sampling_client(model_path=model_path)
        else:
            print(f"Connecting to Tinker base model: {base_model}")
            self.service_client = tinker.ServiceClient()
            self.sampling_client = self.service_client.create_sampling_client(base_model=base_model)
        print("Ready!")

    def chat_completion(self, messages: list, temperature: float = 1.0,
                        max_tokens: int = 4096, **kwargs) -> dict:
        # Disable thinking mode for models that support it (e.g., Qwen3)
        try:
            tokens = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            tokens = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        if hasattr(tokens, "input_ids"):
            tokens = tokens["input_ids"]
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if not isinstance(tokens, list):
            tokens = list(tokens)

        model_input = tinker.ModelInput.from_ints(tokens)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
        )

        result = self.sampling_client.sample(
            prompt=model_input, sampling_params=params, num_samples=1
        ).result()

        output_tokens = result.sequences[0].tokens
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(output_tokens),
                "total_tokens": len(tokens) + len(output_tokens),
            },
        }


def make_handler(proxy: TinkerProxy):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                content_length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_length))
                try:
                    result = proxy.chat_completion(
                        messages=body.get("messages", []),
                        temperature=body.get("temperature", 1.0),
                        max_tokens=body.get("max_tokens", 4096),
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": {"message": str(e), "type": "server_error"}}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": proxy.model_name, "object": "model", "owned_by": "tinker"}]}).encode())
            elif self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            if "POST" in str(args):
                print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible proxy for Tinker models")
    parser.add_argument("--model-path", default=None, help="Tinker FT weights path (tinker://UUID:...). Mutually exclusive with --base-model.")
    parser.add_argument("--base-model", default=None, help="Tinker base model ID (e.g. Qwen/Qwen3-32B). Mutually exclusive with --model-path.")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer ID")
    parser.add_argument("--model-name", default="tinker-ft", help="Model name in API responses")
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    proxy = TinkerProxy(args.model_path, args.tokenizer, args.model_name,
                        base_model=args.base_model)
    handler = make_handler(proxy)

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer((args.host, args.port), handler)
    print(f"\nTinker proxy running at http://{args.host}:{args.port}")
    print(f"Model: {args.model_name}")
    print(f"Use: OPENAI_API_BASE=http://localhost:{args.port}/v1 python run_bloom.py --model {args.model_name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy.")
        server.shutdown()


if __name__ == "__main__":
    main()
