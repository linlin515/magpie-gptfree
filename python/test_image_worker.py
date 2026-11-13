"""Unit tests for ChatGPT Web image worker, PoW, Turnstile, and stdlib PNG mask handling."""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import unittest
import zlib
from pathlib import Path
import sys

PY_DIR = Path(__file__).resolve().parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import image_worker
from utils.turnstile import _xor_string, solve_turnstile_token
from utils.web_pow import _pow_generate, build_legacy_requirements_token, build_pow_config, build_proof_token, parse_pow_resources


def _make_png(width: int, height: int, color_type: int, scanlines: list[bytes], *, bit_depth: int = 8, plte: bytes = b"", trns: bytes = b"") -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    raw = b"".join(scanlines)
    parts = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", ihdr)]
    if plte:
        parts.append(chunk(b"PLTE", plte))
    if trns:
        parts.append(chunk(b"tRNS", trns))
    parts.append(chunk(b"IDAT", zlib.compress(raw)))
    parts.append(chunk(b"IEND", b""))
    return b"".join(parts)


def _make_rgba_png(width: int, height: int, pixels: list[tuple[int, int, int, int]]) -> bytes:
    scanlines = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            r, g, b, a = pixels[y * width + x]
            row.extend((r, g, b, a))
        scanlines.append(bytes(row))
    return _make_png(width, height, 6, scanlines)


def _make_minimal_jpeg(width: int, height: int) -> bytes:
    sof0 = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8\xff\xe0\x00\x04\x00\x00\xff\xc0" + struct.pack(">H", len(sof0) + 2) + sof0 + b"\xff\xd9"


def _make_minimal_webp_vp8x(width: int, height: int) -> bytes:
    w_m1 = (width - 1).to_bytes(3, "little")
    h_m1 = (height - 1).to_bytes(3, "little")
    vp8x_payload = b"\x00\x00\x00\x00" + w_m1 + h_m1
    riff_payload = b"WEBP" + b"VP8X" + struct.pack("<I", len(vp8x_payload)) + vp8x_payload
    return b"RIFF" + struct.pack("<I", len(riff_payload)) + riff_payload


def _make_minimal_gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


class FakeResponse:
    def __init__(self, *, status_code: int = 200, text: str = "", json_data=None, content: bytes = b"", headers=None, lines=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data
        self.content = content
        self.headers = headers or {}
        self._lines = lines or []
        self.closed = False

    def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self.text)

    def iter_lines(self):
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.headers = {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.handler("GET", url, kwargs)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.handler("POST", url, kwargs)

    def put(self, url, **kwargs):
        self.calls.append(("PUT", url, kwargs))
        return self.handler("PUT", url, kwargs)

    def patch(self, url, **kwargs):
        self.calls.append(("PATCH", url, kwargs))
        return self.handler("PATCH", url, kwargs)

    def close(self):
        pass


class ImageWorkerUnitTests(unittest.TestCase):
    def test_web_pow_parses_html_and_solves_sha3_512(self):
        html = '<html data-build="prod-123"><script src="https://cdn.oaistatic.com/c/abc/_app.js"></script></html>'
        scripts, data_build = parse_pow_resources(html)
        self.assertEqual(scripts, ["https://cdn.oaistatic.com/c/abc/_app.js"])
        self.assertEqual(data_build, "c/abc/_")

        p_tok = build_legacy_requirements_token("UA/1.0", scripts, data_build)
        self.assertTrue(p_tok.startswith("gAAAAAC"))

        seed = "test-seed-123"
        difficulty = "0fff"
        proof = build_proof_token(seed, difficulty, "UA/1.0", scripts, data_build, limit=200000)
        self.assertTrue(proof.startswith("gAAAAAB"))
        encoded_answer = proof[len("gAAAAAB") :].encode()
        digest = hashlib.sha3_512(seed.encode() + encoded_answer).digest()
        self.assertLessEqual(digest[:2], bytes.fromhex(difficulty))

    def test_turnstile_solver_executes_xor_bytecode_program(self):
        key = "secret-p-token"
        # Bytecode: [func_2 reg 20 "ok", func_3 reg20 (actually func_3 takes string)]
        # func_19 encodes reg; func_3 expects literal string argument -> wait check solve_turnstile_token signature:
        # token[0] == 3 calls func_3(e) where e is string -> result = base64(e)
        # token[0] == 6 sets reg from tv.nv; test both program decoding + func_3
        program = [[6, 100, 10, 10], [3, "turnstile-passed"]]
        encrypted = _xor_string(json.dumps(program), key)
        dx = base64.b64encode(encrypted.encode()).decode()
        solved = solve_turnstile_token(dx, key)
        self.assertEqual(solved, base64.b64encode(b"turnstile-passed").decode())

    def test_detect_image_info_for_png_jpeg_webp_gif(self):
        png = _make_rgba_png(12, 8, [(255, 0, 0, 255)] * (12 * 8))
        self.assertEqual(image_worker.detect_image_info(png), ("PNG", "image/png", 12, 8))

        jpg = _make_minimal_jpeg(640, 480)
        self.assertEqual(image_worker.detect_image_info(jpg), ("JPEG", "image/jpeg", 640, 480))

        webp = _make_minimal_webp_vp8x(320, 240)
        self.assertEqual(image_worker.detect_image_info(webp), ("WEBP", "image/webp", 320, 240))

        gif = _make_minimal_gif(160, 90)
        self.assertEqual(image_worker.detect_image_info(gif), ("GIF", "image/gif", 160, 90))

    def test_composite_png_mask_handles_filters_and_resizing(self):
        # 2x2 RGBA base image, filter 0
        img = _make_rgba_png(2, 2, [
            (10, 20, 30, 255),
            (40, 50, 60, 255),
            (70, 80, 90, 255),
            (100, 110, 120, 255),
        ])
        # 2x2 mask using Sub filter (type 1) on RGB (color_type=2), luminance -> alpha
        # Pixel (0,0) = 0, (1,0) = 255 -> Sub encoded byte diff
        mask = _make_png(2, 2, 2, [
            bytes([0, 0, 0, 0, 255, 255, 255]),
            bytes([1, 0, 0, 0, 255, 255, 255]),
        ])
        out = image_worker.composite_png_mask(img, mask)
        w, h, ct, rgba = image_worker._decode_png_rgba(out)
        self.assertEqual((w, h, ct), (2, 2, 6))
        self.assertEqual(list(rgba[:4]), [10, 20, 30, 0])
        self.assertEqual(list(rgba[4:8]), [40, 50, 60, 255])

        # Also verify resizing from 1x1 mask (transparent alpha=42) onto 2x2 base image
        small_mask = _make_rgba_png(1, 1, [(0, 0, 0, 42)])
        out2 = image_worker.composite_png_mask(img, small_mask)
        _, _, _, rgba2 = image_worker._decode_png_rgba(out2)
        self.assertEqual(list(rgba2[3::4]), [42, 42, 42, 42])

    def test_image_model_2_6_support_gif_and_rejects_unknown_models(self):
        self.assertEqual(image_worker.normalize_image_model("gpt-image-2.6"), "gpt-image-2.6")
        self.assertEqual(image_worker.normalize_image_model("plus-gpt-image-2.6-flare"), "gpt-image-2.6-flare")
        self.assertEqual(image_worker.normalize_image_model(""), "gpt-image-2")
        self.assertIsNone(image_worker.normalize_image_model("gpt-image-9"))
        self.assertEqual(image_worker.image_model_settings("gpt-image-2")[0], image_worker.DEFAULT_UPSTREAM_MODEL)
        self.assertEqual(image_worker.image_model_settings("gpt-image-2.6-sunburst"), ("auto", ""))
        self.assertIn("gif 动图", image_worker.build_image_prompt(
            "spin", output_format="gif", frame_count=8, frame_duration_ms=120,
        ))

        # GIF is only valid for the 2.6 family and must be rejected for gpt-image-2.
        bad_gif = image_worker.handle_payload({
            "action": "generate_image",
            "access_token": "tok",
            "model": "gpt-image-2",
            "prompt": "spin",
            "output_format": "gif",
        })
        self.assertFalse(bad_gif["ok"])
        self.assertEqual(bad_gif["http_status"], 400)

        bad_model = image_worker.handle_payload({
            "action": "generate_image",
            "access_token": "tok",
            "model": "gpt-image-9",
            "prompt": "x",
        })
        self.assertFalse(bad_model["ok"])
        self.assertEqual(bad_model["error_code"], "model_not_found")

    def test_end_to_end_generation_sse_ignores_user_upload_and_cleans_up(self):
        output_png = _make_rgba_png(4, 4, [(0, 128, 255, 255)] * 16)
        input_png = _make_rgba_png(2, 2, [(255, 255, 255, 255)] * 4)
        mask_png = _make_rgba_png(2, 2, [(0, 0, 0, 0), (255, 255, 255, 255)] * 2)

        def handler(method, url, kwargs):
            if method == "POST" and url == "https://chatgpt.com/backend-api/files":
                return FakeResponse(json_data={"file_id": "file_input_upload_1", "upload_url": "https://upload.example.com/blob1"})
            if method == "PUT" and url == "https://upload.example.com/blob1":
                return FakeResponse(status_code=201)
            if method == "POST" and url == "https://chatgpt.com/backend-api/files/file_input_upload_1/uploaded":
                return FakeResponse(json_data={})
            if method == "GET" and url == "https://chatgpt.com/":
                return FakeResponse(text='<script src="https://cdn.oaistatic.com/c/x/_a.js"></script>')
            if method == "POST" and url.endswith("/sentinel/chat-requirements/prepare"):
                return FakeResponse(json_data={
                    "prepare_token": "prep-1",
                    "proofofwork": {"required": True, "seed": "s1", "difficulty": "0fff"},
                    "turnstile": {"required": False},
                })
            if method == "POST" and url.endswith("/sentinel/chat-requirements/finalize"):
                return FakeResponse(json_data={"token": "sentinel-final-tok"})
            if method == "POST" and url.endswith("/f/conversation/prepare"):
                self.assertEqual(kwargs["json"]["system_hints"], ["picture_v2"])
                return FakeResponse(json_data={"conduit_token": "cond-1"})
            if method == "POST" and url.endswith("/f/conversation"):
                sse_lines = [
                    'data: {"conversation_id":"conv-1","message":{"author":{"role":"user"},"content":{"content_type":"multimodal_text","parts":[{"asset_pointer":"file-service://file_input_upload_1"}]}}}',
                    'data: {"type":"server_ste_metadata","metadata":{"tool_invoked":true,"turn_use_case":"image gen"}}',
                    'data: {"conversation_id":"conv-1","message":{"author":{"role":"tool"},"metadata":{"async_task_type":"image_gen"},"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file_00000000111122223333444455556666"}]}}}',
                    "data: [DONE]",
                ]
                return FakeResponse(lines=sse_lines)
            if method == "GET" and url.endswith("/backend-api/files/file_input_upload_1/download"):
                raise AssertionError("Must NOT download user uploaded input file id")
            if method == "GET" and url.endswith("/backend-api/files/file_00000000111122223333444455556666/download"):
                return FakeResponse(json_data={"download_url": "https://files.oaiusercontent.com/out.png"})
            if method == "GET" and url == "https://files.oaiusercontent.com/out.png":
                hdrs = kwargs.get("headers") or {}
                self.assertNotIn("Authorization", hdrs)
                self.assertNotIn("Origin", hdrs)
                return FakeResponse(content=output_png)
            if method == "PATCH" and url == "https://chatgpt.com/backend-api/conversation/conv-1":
                self.assertEqual(kwargs["json"], {"is_visible": False})
                return FakeResponse(json_data={"success": True})
            raise AssertionError(f"Unexpected HTTP call: {method} {url}")

        session = FakeSession(handler)
        res = image_worker.handle_payload(
            {
                "action": "generate_image",
                "access_token": "tok-abc",
                "prompt": "make the sky blue",
                "size": "1024x1024",
                "quality": "high",
                "output_format": "png",
                "images": [{"b64": base64.b64encode(input_png).decode(), "filename": "orig.png"}],
                "masks": [{"b64": base64.b64encode(mask_png).decode(), "filename": "mask.png"}],
                "settle_secs": 0,
                "initial_wait_secs": 0,
                "poll_interval_secs": 0.01,
            },
            session=session,
        )
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["media_type"], "image/png")
        self.assertEqual(base64.b64decode(res["data"][0]["b64_json"]), output_png)
        self.assertTrue(any(m == "PATCH" and u.endswith("/conv-1") for m, u, _ in session.calls))

    def test_async_polling_and_prompt_recovery_when_sse_short(self):
        output_png = _make_rgba_png(2, 2, [(1, 2, 3, 255)] * 4)
        poll_count = 0

        def handler(method, url, kwargs):
            nonlocal poll_count
            if method == "GET" and url == "https://chatgpt.com/":
                return FakeResponse(text="<html></html>")
            if method == "POST" and url.endswith("/prepare"):
                if "sentinel" in url:
                    return FakeResponse(json_data={"prepare_token": "p", "proofofwork": {"required": False}, "turnstile": {"required": False}})
                return FakeResponse(json_data={"conduit_token": "c"})
            if method == "POST" and url.endswith("/finalize"):
                return FakeResponse(json_data={"token": "t"})
            if method == "POST" and url.endswith("/f/conversation"):
                return FakeResponse(lines=[
                    'data: {"message":{"author":{"role":"assistant"},"channel":"final","content":{"content_type":"text","parts":["Generating {\\"size\\":\\"1024x1024\\",\\"n\\":1}"]}}}',
                    "data: [DONE]",
                ])
            if method == "GET" and "/backend-api/conversations" in url:
                return FakeResponse(json_data={
                    "items": [{"id": "conv-rec-1", "title": "Image cute dog", "update_time": 9999999999.0}]
                })
            if method == "GET" and url.endswith("/backend-api/tasks"):
                return FakeResponse(json_data={"tasks": []})
            if method == "GET" and url == "https://chatgpt.com/backend-api/conversation/conv-rec-1":
                poll_count += 1
                if poll_count == 1:
                    return FakeResponse(json_data={"mapping": {}})
                return FakeResponse(json_data={
                    "mapping": {
                        "n1": {
                            "message": {
                                "author": {"role": "tool"},
                                "metadata": {"async_task_type": "image_gen"},
                                "content": {"parts": [{"content_type": "image_asset_pointer", "asset_pointer": "sediment://file_sed_99"}]},
                            }
                        }
                    }
                })
            if method == "GET" and url.endswith("/attachment/file_sed_99/download"):
                return FakeResponse(json_data={"download_url": "https://files.oaiusercontent.com/sed.png"})
            if method == "GET" and url == "https://files.oaiusercontent.com/sed.png":
                return FakeResponse(content=output_png)
            if method == "PATCH":
                return FakeResponse(json_data={})
            raise AssertionError(f"Unexpected call: {method} {url}")

        session = FakeSession(handler)
        res = image_worker.handle_payload(
            {
                "action": "generate_image",
                "access_token": "tok",
                "prompt": "cute dog",
                "initial_wait_secs": 0,
                "poll_interval_secs": 0.01,
                "settle_secs": 0,
                "poll_timeout_secs": 5,
            },
            session=session,
        )
        self.assertTrue(res["ok"])
        self.assertEqual(poll_count, 2)
        self.assertEqual(base64.b64decode(res["data"][0]["b64_json"]), output_png)

    def test_error_mapping_401_400_policy_429_and_arkose_503_without_leaks(self):
        # 1. HTTP 401 -> unauthorized
        res_401 = image_worker.handle_payload(
            {"action": "generate_image", "access_token": "eyJhbGci.secret.sig", "prompt": "cat"},
            session=FakeSession(lambda m, u, k: FakeResponse(status_code=401, text='{"detail":"token_invalidated user@example.com"}')),
        )
        self.assertEqual(res_401["http_status"], 401)
        self.assertEqual(res_401["error_code"], "unauthorized")
        self.assertNotIn("eyJhbGci", json.dumps(res_401))
        self.assertNotIn("user@example.com", json.dumps(res_401))

        # 2. HTTP 429 -> rate_limited
        res_429 = image_worker.handle_payload(
            {"action": "generate_image", "access_token": "tok", "prompt": "cat"},
            session=FakeSession(lambda m, u, k: FakeResponse(status_code=429, text="slow down", headers={"Retry-After": "12"})),
        )
        self.assertEqual(res_429["http_status"], 429)
        self.assertEqual(res_429["retry_after"], 12)

        # 3. Arkose -> 503 failover
        def arkose_handler(m, u, k):
            if m == "GET":
                return FakeResponse(text="<html></html>")
            return FakeResponse(json_data={"arkose": {"required": True}})

        res_arkose = image_worker.handle_payload(
            {"action": "generate_image", "access_token": "tok", "prompt": "cat"},
            session=FakeSession(arkose_handler),
        )
        self.assertEqual(res_arkose["http_status"], 503)
        self.assertEqual(res_arkose["error_code"], "arkose_required")

        # 4. Content policy refusal -> 400 content_policy_violation
        def policy_handler(m, u, k):
            if m == "GET" and u == "https://chatgpt.com/":
                return FakeResponse(text="<html></html>")
            if m == "POST" and "prepare" in u:
                return FakeResponse(json_data={"prepare_token": "p", "conduit_token": "c"})
            if m == "POST" and "finalize" in u:
                return FakeResponse(json_data={"token": "t"})
            if m == "POST" and u.endswith("/f/conversation"):
                return FakeResponse(lines=[
                    'data: {"conversation_id":"conv-p","message":{"author":{"role":"assistant"},"channel":"final","content":{"content_type":"text","parts":["抱歉，该请求违反了我们的内容政策，无法生成。"]}}}',
                    "data: [DONE]",
                ])
            if m == "GET" and u.endswith("/backend-api/tasks"):
                return FakeResponse(json_data={"tasks": []})
            if m == "PATCH":
                return FakeResponse(json_data={})
            raise AssertionError(f"Unexpected call: {m} {u}")

        res_policy = image_worker.handle_payload(
            {"action": "generate_image", "access_token": "tok", "prompt": "unsafe", "initial_wait_secs": 0, "settle_secs": 0},
            session=FakeSession(policy_handler),
        )
        self.assertEqual(res_policy["http_status"], 400)
        self.assertEqual(res_policy["error_code"], "content_policy_violation")

    def test_estuary_download_isolates_headers_and_deduplicates_same_asset_id(self):
        output_png = _make_rgba_png(2, 2, [(9, 8, 7, 255)] * 4)
        estuary_url = "https://chatgpt.com/backend-api/estuary/content?id=file_00000000aaaabbbbccccdddd11112222&sig=abc"
        seen_urls: list[str] = []

        def handler(method, url, kwargs):
            seen_urls.append(f"{method} {url}")
            if method == "GET" and url.endswith("/backend-api/files/file_00000000aaaabbbbccccdddd11112222/download"):
                return FakeResponse(json_data={"download_url": estuary_url})
            if method == "GET" and "/attachment/" in url:
                raise AssertionError("Should not double-query attachment endpoint for same already-resolved asset id")
            if method == "GET" and url == estuary_url:
                hdrs = kwargs.get("headers") or {}
                self.assertEqual(hdrs.get("Authorization"), "Bearer tok-estuary")
                self.assertEqual(hdrs.get("Referer"), "https://chatgpt.com/")
                self.assertNotIn("Origin", hdrs)
                self.assertNotIn("OAI-Device-Id", hdrs)
                self.assertNotIn("Sec-Fetch-Mode", hdrs)
                return FakeResponse(content=output_png)
            raise AssertionError(f"Unexpected call: {method} {url}")

        session = FakeSession(handler)
        with image_worker.OpenAIWebImageClient("tok-estuary", session=session) as client:
            urls = client._resolve_image_urls(
                "conv-est",
                ["file_00000000aaaabbbbccccdddd11112222"],
                ["file_00000000aaaabbbbccccdddd11112222"],
            )
            self.assertEqual(urls, [estuary_url])
            imgs = client._download_image_bytes(urls)
            self.assertEqual(imgs, [output_png])

    def test_check_image_quota_parses_limits_progress_plan_and_rate_limits(self):
        # 1. Active account with image_gen remaining quota and reset_after
        def quota_handler(method, url, kwargs):
            if method == "POST" and url == "https://chatgpt.com/backend-api/conversation/init":
                return FakeResponse(json_data={
                    "default_model_slug": "auto",
                    "limits_progress": [
                        {"feature_name": "other_tool", "remaining": 5},
                        {"feature_name": "image_gen", "remaining": 18, "reset_after": "2026-11-10T12:00:00+00:00"},
                    ],
                })
            if method == "GET" and url.startswith("https://chatgpt.com/backend-api/accounts/check/v4-2023-04-27"):
                return FakeResponse(json_data={
                    "accounts": {"default": {"account": {"plan_type": "free"}}},
                })
            raise AssertionError(f"Unexpected call: {method} {url}")

        res = image_worker.handle_payload(
            {"action": "check_image_quota", "access_token": "tok-q1"},
            session=FakeSession(quota_handler),
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["quota"], 18)
        self.assertEqual(res["restore_at"], "2026-11-10T12:00:00+00:00")
        self.assertTrue(res["has_image_limit"])
        self.assertEqual(res["plan_type"], "free")
        self.assertEqual(res["default_model_slug"], "auto")

        # 2. Missing image_gen entry defaults to quota 0 and keeps working even if accounts/check fails
        def empty_limits_handler(method, url, kwargs):
            if method == "POST" and url.endswith("/conversation/init"):
                return FakeResponse(json_data={"limits_progress": []})
            return FakeResponse(status_code=500, text="temporary check error")

        res_empty = image_worker.handle_payload(
            {"action": "check_image_quota", "access_token": "tok-q2"},
            session=FakeSession(empty_limits_handler),
        )
        self.assertTrue(res_empty["ok"])
        self.assertEqual(res_empty["quota"], 0)
        self.assertFalse(res_empty["has_image_limit"])
        self.assertIsNone(res_empty["restore_at"])

        # 3. HTTP 401 and JSON 429 retry_after extraction
        res_unauth = image_worker.handle_payload(
            {"action": "check_image_quota", "access_token": "tok-bad"},
            session=FakeSession(lambda m, u, k: FakeResponse(status_code=401, json_data={"detail": "token_invalidated"})),
        )
        self.assertFalse(res_unauth["ok"])
        self.assertEqual(res_unauth["http_status"], 401)
        self.assertEqual(res_unauth["error_code"], "unauthorized")

        res_429 = image_worker.handle_payload(
            {"action": "check_image_quota", "access_token": "tok-429"},
            session=FakeSession(lambda m, u, k: FakeResponse(status_code=429, json_data={"detail": {"clears_in": 95}})),
        )
        self.assertFalse(res_429["ok"])
        self.assertEqual(res_429["http_status"], 429)
        self.assertEqual(res_429["retry_after"], 95)


    def test_check_token_uses_accounts_check_and_maps_statuses(self):
        def ok_handler(method, url, kwargs):
            if method == "GET" and url.startswith("https://chatgpt.com/backend-api/accounts/check/v4-2023-04-27"):
                return FakeResponse(json_data={
                    "accounts": {"default": {"account": {"plan_type": "free"}}},
                })
            raise AssertionError(f"Unexpected call: {method} {url}")

        res = image_worker.handle_payload(
            {"action": "check_token", "access_token": "tok-web"},
            session=FakeSession(ok_handler),
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["http_status"], 200)
        self.assertEqual(res["plan_type"], "free")

        res_401 = image_worker.handle_payload(
            {"action": "check_token", "access_token": "tok-bad"},
            session=FakeSession(lambda m, u, k: FakeResponse(status_code=401, json_data={"detail": "token_invalidated"})),
        )
        self.assertFalse(res_401["ok"])
        self.assertEqual(res_401["http_status"], 401)
        self.assertEqual(res_401["error_code"], "unauthorized")

        res_403 = image_worker.handle_payload(
            {"action": "check_token", "access_token": "tok-cf"},
            session=FakeSession(lambda m, u, k: FakeResponse(status_code=403, text="blocked")),
        )
        self.assertFalse(res_403["ok"])
        self.assertEqual(res_403["http_status"], 403)
        self.assertEqual(res_403["error_code"], "forbidden")


if __name__ == "__main__":
    unittest.main()
