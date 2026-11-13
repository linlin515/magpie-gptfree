"""Standalone ChatGPT Web picture_v2 image generation & editing worker.

Adapted from chatgpt2api-native (OpenAIBackendAPI + conversation protocol)
using only curl_cffi + Python standard library (zero Pillow / FastAPI / pybase64).
One request per process: reads JSON on stdin, writes sanitized JSON on stdout.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import socket
import struct
import sys
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from curl_cffi import requests
except ImportError:
    requests = None

from utils.turnstile import solve_turnstile_token
from utils.web_pow import (
    DEFAULT_POW_SCRIPT,
    build_legacy_requirements_token,
    build_proof_token,
    parse_pow_resources,
)

DEFAULT_CLIENT_VERSION = "prod-a194cd50d4416d3c0b47c740f206b12ce60f5887"
DEFAULT_CLIENT_BUILD_NUMBER = "6708908"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
)
DEFAULT_SEC_CH_UA = '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"'

MAX_STDIN_BYTES = 48 * 1024 * 1024
MAX_INPUT_IMAGES = 10
MAX_INPUT_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PNG_DECODED_BYTES = 100 * 1024 * 1024
SUPPORTED_OUTPUT_FORMATS = {"", "auto", "png", "webp", "jpeg", "jpg", "gif"}

# Models exposed to clients (mirrors yang0/chatgpt2api-dsh).
IMAGE_MODELS = (
    "gpt-image-2",
    "gpt-image-2.6",
    "gpt-image-2.6-flare",
    "gpt-image-2.6-sunburst",
)
GIF_CAPABLE_MODELS = {"gpt-image-2.6", "gpt-image-2.6-flare", "gpt-image-2.6-sunburst"}
IMAGE_MODEL_PLAN_TYPES = ("plus", "team", "pro")
DEFAULT_UPSTREAM_MODEL = "gpt-5-5"


def normalize_image_model(model: Any) -> str | None:
    """Map a client model name (possibly plan-prefixed) to a supported base model."""
    raw = str(model or "").strip().lower()
    if not raw:
        return "gpt-image-2"
    if raw in IMAGE_MODELS:
        return raw
    for plan_type in IMAGE_MODEL_PLAN_TYPES:
        prefix = f"{plan_type}-"
        if raw.startswith(prefix) and raw[len(prefix):] in IMAGE_MODELS:
            return raw[len(prefix):]
    return None


def image_model_settings(model: Any) -> tuple[str, str]:
    """Return (upstream_model, thinking_effort) for a client-facing image model."""
    base_model = normalize_image_model(model)
    if base_model == "gpt-image-2":
        return DEFAULT_UPSTREAM_MODEL, ""
    if base_model in GIF_CAPABLE_MODELS:
        # The 2.6 family lets the web client pick the model itself.
        return "auto", ""
    return "auto", ""

FILE_SERVICE_ID_RE = re.compile(r"file-service://([A-Za-z0-9_-]+)")
REAL_IMAGE_FILE_ID_RE = re.compile(r"\bfile_00000000[a-f0-9]{24}\b")
SEDIMENT_ID_RE = re.compile(r"sediment://([A-Za-z0-9_-]+)")
REFERENCED_IMAGE_IDS_RE = re.compile(r'"referenced_image_ids"\s*:\s*\[([^\]]+)\]')
TOOL_PARAMS_JSON_RE = re.compile(r'\{\s*"size"\s*:\s*"\d+x\d+"\s*,\s*"n"\s*:\s*\d+\s*\}')

_CONTENT_POLICY_KEYWORDS = (
    "内容政策",
    "防护限制",
    "违反",
    "moderation",
    "policy",
    "blocked",
    "不能生成",
    "无法生成",
    "不能帮助",
    "无法帮助",
    "裸体",
    "裸露",
    "色情",
    "性内容",
    "未成年",
    "抱歉，我不能",
    "content policy",
    "safety system",
)


class ImageWorkerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        http_status: int = 502,
        error_code: str = "upstream_error",
        conversation_id: str = "",
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.error_code = error_code
        self.conversation_id = conversation_id
        self.retry_after = retry_after


class ImageContentPolicyError(ImageWorkerError):
    def __init__(self, message: str, conversation_id: str = "") -> None:
        super().__init__(
            message or "Image generation was rejected by upstream content policy.",
            http_status=400,
            error_code="content_policy_violation",
            conversation_id=conversation_id,
        )


class ImagePollTimeoutError(ImageWorkerError):
    def __init__(
        self,
        message: str,
        conversation_id: str = "",
        task_error: str = "",
    ) -> None:
        super().__init__(
            message,
            http_status=504,
            error_code="timeout",
            conversation_id=conversation_id,
        )
        self.task_error = task_error


class ImageStreamHardTimeoutError(ImageWorkerError):
    def __init__(self, message: str, conversation_id: str = "") -> None:
        super().__init__(
            message,
            http_status=504,
            error_code="stream_timeout",
            conversation_id=conversation_id,
        )


class UpstreamHTTPError(RuntimeError):
    def __init__(
        self,
        context: str,
        status_code: int,
        body: Any = "",
        retry_after: int | None = None,
    ) -> None:
        self.context = context
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        super().__init__(f"{context} failed: HTTP {status_code}")


@dataclass
class ChatRequirements:
    token: str
    proof_token: str = ""
    turnstile_token: str = ""
    so_token: str = ""
    raw_finalize: Optional[Dict[str, Any]] = None


@dataclass
class ConversationState:
    text: str = ""
    raw_text: str = ""
    conversation_id: str = ""
    file_ids: list[str] = field(default_factory=list)
    sediment_ids: list[str] = field(default_factory=list)
    blocked: bool = False
    tool_invoked: bool | None = None
    turn_use_case: str = ""


def resolve_proxy(custom_proxy: str = "") -> str:
    if custom_proxy and str(custom_proxy).strip():
        return str(custom_proxy).strip()
    for env_key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    for port in (7890, 7897, 10809, 10808):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.15):
                return f"http://127.0.0.1:{port}"
        except OSError:
            continue
    return ""


def _sanitize_error_text(text: str, limit: int = 300) -> str:
    s = str(text or "").strip()
    if not s:
        return "Image generation failed"
    s = re.sub(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]+)?", "[redacted-token]", s)
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", s)
    s = re.sub(r"https?://[^\s\"'<>]+", "[url]", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "Image generation failed"
    return s[:limit]


def _is_token_invalid_text(text: str) -> bool:
    low = str(text or "").lower()
    return (
        "token_invalidated" in low
        or "token_revoked" in low
        or "authentication token has been invalidated" in low
        or "invalidated oauth token" in low
    )


def _is_content_policy_error(error_msg: str) -> bool:
    if not error_msg:
        return False
    low = error_msg.lower()
    return any(keyword in low for keyword in _CONTENT_POLICY_KEYWORDS)


def _is_model_text_reply_instead_of_image(message: str) -> bool:
    if not message:
        return False
    return bool(REFERENCED_IMAGE_IDS_RE.search(message) or TOOL_PARAMS_JSON_RE.search(message))


def _validate_https_url(url: str, label: str = "url") -> str:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ImageWorkerError(
            f"Invalid {label}: expected https URL",
            http_status=502,
            error_code="upstream_error",
        )
    return str(url).strip()


# ---------------------------------------------------------------------------
# Pure-stdlib Image Inspection & PNG Alpha Mask Compositing (no Pillow)
# ---------------------------------------------------------------------------

def sniff_image_mime(data: bytes, fallback: str = "image/png") -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return fallback


def detect_image_info(data: bytes) -> tuple[str, str, int, int]:
    """Return (format, mime_type, width, height) using stdlib only."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 10:
        raise ImageWorkerError("Invalid or empty image payload", http_status=400, error_code="invalid_image")
    raw = bytes(data)

    # PNG
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24 and raw[12:16] == b"IHDR":
        width, height = struct.unpack(">II", raw[16:24])
        if width > 0 and height > 0:
            return "PNG", "image/png", width, height

    # GIF
    if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
        width, height = struct.unpack("<HH", raw[6:10])
        if width > 0 and height > 0:
            return "GIF", "image/gif", width, height

    # WEBP
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" and len(raw) >= 30:
        chunk = raw[12:16]
        if chunk == b"VP8 " and raw[23:26] == b"\x9d\x01\x2a":
            width = struct.unpack("<H", raw[26:28])[0] & 0x3FFF
            height = struct.unpack("<H", raw[28:30])[0] & 0x3FFF
            if width > 0 and height > 0:
                return "WEBP", "image/webp", width, height
        elif chunk == b"VP8L" and raw[20] == 0x2F:
            bits = struct.unpack("<I", raw[21:25])[0]
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            if width > 0 and height > 0:
                return "WEBP", "image/webp", width, height
        elif chunk == b"VP8X":
            width = 1 + int.from_bytes(raw[24:27], "little")
            height = 1 + int.from_bytes(raw[27:30], "little")
            if width > 0 and height > 0:
                return "WEBP", "image/webp", width, height

    # JPEG
    if raw[:2] == b"\xff\xd8":
        pos = 2
        n = len(raw)
        while pos < n:
            while pos < n and raw[pos] != 0xFF:
                pos += 1
            while pos < n and raw[pos] == 0xFF:
                pos += 1
            if pos >= n:
                break
            marker = raw[pos]
            pos += 1
            if marker in (0x00, 0x01, 0xD8) or (0xD0 <= marker <= 0xD7):
                continue
            if marker in (0xD9, 0xDA):
                break
            if pos + 2 > n:
                break
            seg_len = struct.unpack(">H", raw[pos : pos + 2])[0]
            if seg_len < 2 or pos + seg_len > n:
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if pos + 7 <= n:
                    height, width = struct.unpack(">HH", raw[pos + 3 : pos + 7])
                    if width > 0 and height > 0:
                        return "JPEG", "image/jpeg", width, height
                break
            pos += seg_len

    raise ImageWorkerError("Unsupported or malformed image data", http_status=400, error_code="invalid_image")


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _decode_png_rgba(data: bytes) -> tuple[int, int, int, bytearray]:
    """Decode non-interlaced PNG into (width, height, color_type, rgba_bytearray)."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageWorkerError("Mask compositing requires PNG images", http_status=400, error_code="invalid_image")
    pos = 8
    width = height = bit_depth = color_type = 0
    ihdr_seen = False
    plte = b""
    trns = b""
    idat_parts: list[bytes] = []

    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        pos += 8
        if pos + length + 4 > len(data):
            raise ImageWorkerError("Truncated PNG chunk", http_status=400, error_code="invalid_image")
        chunk_data = data[pos : pos + length]
        pos += length + 4
        if ctype == b"IHDR":
            if length != 13:
                raise ImageWorkerError("Invalid PNG IHDR", http_status=400, error_code="invalid_image")
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk_data)
            if comp != 0 or filt != 0:
                raise ImageWorkerError("Unsupported PNG compression or filter", http_status=400, error_code="invalid_image")
            if interlace != 0:
                raise ImageWorkerError("Interlaced PNG images are not supported with mask", http_status=400, error_code="invalid_image")
            if color_type not in (0, 2, 3, 4, 6) or bit_depth not in (1, 2, 4, 8, 16):
                raise ImageWorkerError("Unsupported PNG format", http_status=400, error_code="invalid_image")
            ihdr_seen = True
        elif ctype == b"PLTE":
            plte = chunk_data
        elif ctype == b"tRNS":
            trns = chunk_data
        elif ctype == b"IDAT":
            idat_parts.append(chunk_data)
        elif ctype == b"IEND":
            break

    if not ihdr_seen or width <= 0 or height <= 0 or not idat_parts:
        raise ImageWorkerError("Malformed PNG image", http_status=400, error_code="invalid_image")

    samples_per_pixel = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    bits_per_pixel = samples_per_pixel * bit_depth
    bpp = max(1, (bits_per_pixel + 7) // 8)
    scanline_len = (width * bits_per_pixel + 7) // 8
    expected_bytes = height * (1 + scanline_len)
    if expected_bytes > MAX_PNG_DECODED_BYTES:
        raise ImageWorkerError("PNG image is too large to composite", http_status=400, error_code="invalid_image")

    try:
        raw = zlib.decompress(b"".join(idat_parts))
    except zlib.error as exc:
        raise ImageWorkerError("Corrupt PNG IDAT stream", http_status=400, error_code="invalid_image") from exc

    if len(raw) < expected_bytes:
        raise ImageWorkerError("Incomplete PNG image data", http_status=400, error_code="invalid_image")

    # Unfilter scanlines
    rows: list[bytearray] = []
    prev = bytearray(scanline_len)
    offset = 0
    for _ in range(height):
        filt_type = raw[offset]
        offset += 1
        cur = bytearray(raw[offset : offset + scanline_len])
        offset += scanline_len
        if filt_type == 0:
            pass
        elif filt_type == 1:
            for i in range(scanline_len):
                a = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + a) & 0xFF
        elif filt_type == 2:
            for i in range(scanline_len):
                cur[i] = (cur[i] + prev[i]) & 0xFF
        elif filt_type == 3:
            for i in range(scanline_len):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                cur[i] = (cur[i] + ((a + b) >> 1)) & 0xFF
        elif filt_type == 4:
            for i in range(scanline_len):
                a = cur[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + _paeth_predictor(a, b, c)) & 0xFF
        else:
            raise ImageWorkerError("Unknown PNG scanline filter", http_status=400, error_code="invalid_image")
        rows.append(cur)
        prev = cur

    rgba = bytearray(width * height * 4)
    trans_gray = struct.unpack(">H", trns[:2])[0] if color_type == 0 and len(trns) >= 2 else None
    trans_rgb = struct.unpack(">HHH", trns[:6]) if color_type == 2 and len(trns) >= 6 else None

    out_idx = 0
    for row in rows:
        if bit_depth == 8:
            if color_type == 6:
                rgba[out_idx : out_idx + width * 4] = row[: width * 4]
                out_idx += width * 4
                continue
            for x in range(width):
                if color_type == 0:
                    g = row[x]
                    a = 0 if (trans_gray is not None and g == trans_gray) else 255
                    rgba[out_idx : out_idx + 4] = (g, g, g, a)
                elif color_type == 2:
                    r, g, b = row[x * 3], row[x * 3 + 1], row[x * 3 + 2]
                    a = 0 if (trans_rgb is not None and (r, g, b) == trans_rgb) else 255
                    rgba[out_idx : out_idx + 4] = (r, g, b, a)
                elif color_type == 3:
                    idx = row[x]
                    if idx * 3 + 2 < len(plte):
                        r, g, b = plte[idx * 3], plte[idx * 3 + 1], plte[idx * 3 + 2]
                    else:
                        r = g = b = 0
                    a = trns[idx] if idx < len(trns) else 255
                    rgba[out_idx : out_idx + 4] = (r, g, b, a)
                elif color_type == 4:
                    g, a = row[x * 2], row[x * 2 + 1]
                    rgba[out_idx : out_idx + 4] = (g, g, g, a)
                out_idx += 4
        elif bit_depth == 16:
            step = samples_per_pixel * 2
            for x in range(width):
                base = x * step
                if color_type == 0:
                    val16 = struct.unpack(">H", row[base : base + 2])[0]
                    g = val16 >> 8
                    a = 0 if (trans_gray is not None and val16 == trans_gray) else 255
                    rgba[out_idx : out_idx + 4] = (g, g, g, a)
                elif color_type == 2:
                    r16, g16, b16 = struct.unpack(">HHH", row[base : base + 6])
                    a = 0 if (trans_rgb is not None and (r16, g16, b16) == trans_rgb) else 255
                    rgba[out_idx : out_idx + 4] = (r16 >> 8, g16 >> 8, b16 >> 8, a)
                elif color_type == 4:
                    g = row[base]
                    a = row[base + 2]
                    rgba[out_idx : out_idx + 4] = (g, g, g, a)
                elif color_type == 6:
                    rgba[out_idx : out_idx + 4] = (row[base], row[base + 2], row[base + 4], row[base + 6])
                out_idx += 4
        else:
            # 1, 2, or 4 bit depth (color types 0 or 3)
            max_val = (1 << bit_depth) - 1
            mask = max_val
            per_byte = 8 // bit_depth
            for x in range(width):
                byte_val = row[x // per_byte]
                shift = 8 - bit_depth * ((x % per_byte) + 1)
                sample = (byte_val >> shift) & mask
                if color_type == 3:
                    if sample * 3 + 2 < len(plte):
                        r, g, b = plte[sample * 3], plte[sample * 3 + 1], plte[sample * 3 + 2]
                    else:
                        r = g = b = 0
                    a = trns[sample] if sample < len(trns) else 255
                    rgba[out_idx : out_idx + 4] = (r, g, b, a)
                else:
                    g = (sample * 255 + (max_val // 2)) // max_val
                    a = 0 if (trans_gray is not None and sample == trans_gray) else 255
                    rgba[out_idx : out_idx + 4] = (g, g, g, a)
                out_idx += 4

    return width, height, color_type, rgba


def _encode_png_rgba(width: int, height: int, rgba: bytes | bytearray) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    stride = width * 4
    buf = bytearray((stride + 1) * height)
    for y in range(height):
        dst = y * (stride + 1)
        src = y * stride
        buf[dst] = 0
        buf[dst + 1 : dst + 1 + stride] = rgba[src : src + stride]
    compressed = zlib.compress(bytes(buf), level=6)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


def _resize_alpha_bilinear(alpha: bytes | bytearray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> bytearray:
    if src_w == dst_w and src_h == dst_h:
        return bytearray(alpha)
    out = bytearray(dst_w * dst_h)
    for y in range(dst_h):
        sy = ((y + 0.5) * src_h / dst_h) - 0.5
        sy = max(0.0, min(src_h - 1.0, sy))
        y0 = int(sy)
        y1 = min(y0 + 1, src_h - 1)
        fy = sy - y0
        for x in range(dst_w):
            sx = ((x + 0.5) * src_w / dst_w) - 0.5
            sx = max(0.0, min(src_w - 1.0, sx))
            x0 = int(sx)
            x1 = min(x0 + 1, src_w - 1)
            fx = sx - x0
            a00 = alpha[y0 * src_w + x0]
            a10 = alpha[y0 * src_w + x1]
            a01 = alpha[y1 * src_w + x0]
            a11 = alpha[y1 * src_w + x1]
            val = (
                a00 * (1.0 - fx) * (1.0 - fy)
                + a10 * fx * (1.0 - fy)
                + a01 * (1.0 - fx) * fy
                + a11 * fx * fy
            )
            out[y * dst_w + x] = max(0, min(255, int(round(val))))
    return out


def composite_png_mask(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    """Apply mask's alpha channel (or luminance if no alpha) onto PNG image."""
    img_w, img_h, _, img_rgba = _decode_png_rgba(image_bytes)
    mask_w, mask_h, mask_ct, mask_rgba = _decode_png_rgba(mask_bytes)
    if mask_ct in (4, 6):
        alpha = bytearray(mask_rgba[3::4])
    else:
        alpha = bytearray(mask_w * mask_h)
        for i in range(mask_w * mask_h):
            r, g, b = mask_rgba[i * 4], mask_rgba[i * 4 + 1], mask_rgba[i * 4 + 2]
            alpha[i] = (299 * r + 587 * g + 114 * b + 500) // 1000
    resized = _resize_alpha_bilinear(alpha, mask_w, mask_h, img_w, img_h)
    for i in range(img_w * img_h):
        img_rgba[i * 4 + 3] = resized[i]
    return _encode_png_rgba(img_w, img_h, img_rgba)


def composite_images_with_masks(
    images: list[tuple[bytes, str, str]],
    masks: list[tuple[bytes, str, str]],
) -> list[tuple[bytes, str, str]]:
    if not masks:
        return images
    out: list[tuple[bytes, str, str]] = []
    for idx, (img_bytes, filename, _) in enumerate(images):
        mask_bytes = masks[idx][0] if idx < len(masks) else masks[-1][0]
        composited = composite_png_mask(img_bytes, mask_bytes)
        out_name = filename if filename.lower().endswith(".png") else f"{Path(filename).stem or f'image_{idx + 1}'}.png"
        out.append((composited, out_name, "image/png"))
    return out


def build_image_prompt(
    prompt: str,
    *,
    size: str | None = None,
    quality: str = "auto",
    output_format: str | None = None,
    background: str | None = None,
    frame_count: int | None = None,
    frame_duration_ms: int | None = None,
) -> str:
    hints: list[str] = []
    size_clean = str(size or "").strip()
    if size_clean and size_clean.lower() != "auto":
        hints.append(f"输出图片尺寸为 {size_clean}。")
    quality_clean = str(quality or "").strip()
    if quality_clean and quality_clean.lower() != "auto":
        hints.append(f"输出图片质量为 {quality_clean}。")
    fmt_clean = str(output_format or "").strip().lower()
    if fmt_clean == "jpg":
        fmt_clean = "jpeg"
    if fmt_clean == "gif":
        hints.append("生成格式为 gif 动图；输出真正的多帧动画，不要输出分镜拼图或静态帧。")
        if frame_count is not None:
            hints.append(f"帧数约为 {int(frame_count)} 帧。")
        if frame_duration_ms is not None:
            hints.append(f"每帧时长约为 {int(frame_duration_ms)} 毫秒。")
    elif fmt_clean and fmt_clean != "auto":
        hints.append(f"输出图片格式为 {fmt_clean}。")
    bg_clean = str(background or "").strip().lower()
    if bg_clean and bg_clean != "auto":
        if bg_clean == "transparent":
            hints.append("生成透明背景（PNG 透明通道）。")
        else:
            hints.append(f"背景风格为 {bg_clean}。")
    base = str(prompt or "").strip()
    return f"{base}\n\n{''.join(hints)}" if hints else base


def _iter_sse_payloads(response: Any) -> Iterator[str]:
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload:
            yield payload


def _add_unique(values: list[str], candidates: Sequence[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in values:
            values.append(candidate)


def _extract_conversation_ids(payload: str) -> tuple[str, list[str], list[str]]:
    conversation_match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
    conversation_id = conversation_match.group(1) if conversation_match else ""
    file_ids: list[str] = []
    _add_unique(file_ids, FILE_SERVICE_ID_RE.findall(payload))
    _add_unique(file_ids, REAL_IMAGE_FILE_ID_RE.findall(payload))
    sediment_ids = SEDIMENT_ID_RE.findall(payload)
    return conversation_id, file_ids, sediment_ids


def _is_image_tool_event(event: dict[str, Any]) -> bool:
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata") or {}
    author = message.get("author") or {}
    content = message.get("content") or {}
    if author.get("role") != "tool":
        return False
    if metadata.get("async_task_type") == "image_gen":
        return True
    if content.get("content_type") != "multimodal_text":
        return False
    return any(
        isinstance(part, dict)
        and (
            part.get("content_type") == "image_asset_pointer"
            or str(part.get("asset_pointer") or "").startswith(("file-service://", "sediment://"))
        )
        for part in (content.get("parts") or [])
    )


def _is_user_message_event(event: dict[str, Any]) -> bool:
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if isinstance(message, dict):
        author = message.get("author") or {}
        if str(author.get("role") or "").strip().lower() == "user":
            return True
    return False


def _is_visible_assistant_message(message: dict[str, Any]) -> bool:
    author = message.get("author")
    if not isinstance(author, dict):
        return False
    if str(author.get("role") or "").strip().lower() != "assistant":
        return False
    metadata = message.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("is_visually_hidden_from_conversation") is True:
        return False
    recipient = str(message.get("recipient") or "").strip().lower()
    if recipient and recipient != "all":
        return False
    channel = str(message.get("channel") or "").strip().lower()
    if channel and channel != "final":
        return False
    return True


def _assistant_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    if isinstance(content, str):
        return content
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts") or []
    if isinstance(parts, list) and parts:
        text = "".join(p for p in parts if isinstance(p, str))
        if text:
            return text
    return str(content.get("text") or "")


def _apply_patch_op(op_dict: dict[str, Any], current_text: str) -> str:
    op = op_dict.get("o")
    val = str(op_dict.get("v") or "")
    if op == "append":
        return current_text + val
    if op == "replace":
        return val
    return current_text


def _assistant_raw_text(event: dict[str, Any], current_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if isinstance(message, dict) and _is_visible_assistant_message(message):
            text = _assistant_message_text(message)
            if text:
                return text
    if event.get("p") == "/message/content/parts/0":
        return _apply_patch_op(event, current_text)
    operations = event.get("v")
    if isinstance(operations, str) and current_text and not event.get("p") and not event.get("o"):
        return current_text + operations
    if isinstance(operations, list):
        text = current_text
        for item in operations:
            if isinstance(item, dict) and (item.get("p") == "/message/content/parts/0" or "o" in item):
                if item.get("p") in (None, "/message/content/parts/0", ""):
                    text = _apply_patch_op(item, text)
        return text
    return current_text


def _sanitize_output_text(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"\ue200[^\ue201]*\ue201", "", s)
    s = re.sub(r"\ue200[^\ue201]*$", "", s)
    return s


def _update_conversation_state(state: ConversationState, payload: str, event: dict[str, Any] | None = None) -> None:
    conv_id, file_ids, sediment_ids = _extract_conversation_ids(payload)
    if conv_id and not state.conversation_id:
        state.conversation_id = conv_id

    is_patch_event = isinstance(event, dict) and event.get("o") == "patch"
    is_user_msg = isinstance(event, dict) and _is_user_message_event(event)
    image_context = (
        (isinstance(event, dict) and _is_image_tool_event(event))
        or (state.tool_invoked is True and not is_user_msg)
        or (
            is_patch_event
            and not is_user_msg
            and ("asset_pointer" in payload or "file-service://" in payload or "sediment://" in payload)
        )
    )
    if image_context:
        _add_unique(state.file_ids, file_ids)
        _add_unique(state.sediment_ids, sediment_ids)

    if not isinstance(event, dict):
        return
    state.conversation_id = str(event.get("conversation_id") or state.conversation_id)
    value = event.get("v")
    if isinstance(value, dict):
        state.conversation_id = str(value.get("conversation_id") or state.conversation_id)
    if event.get("type") == "moderation":
        moderation = event.get("moderation_response")
        if isinstance(moderation, dict) and moderation.get("blocked") is True:
            state.blocked = True
    if event.get("type") == "server_ste_metadata":
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            if isinstance(metadata.get("tool_invoked"), bool):
                state.tool_invoked = metadata["tool_invoked"]
            state.turn_use_case = str(metadata.get("turn_use_case") or state.turn_use_case)


def _extract_image_reference_ids(payload: Any) -> tuple[list[str], list[str]]:
    file_ids: list[str] = []
    sediment_ids: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            _add_unique(file_ids, FILE_SERVICE_ID_RE.findall(value))
            _add_unique(file_ids, REAL_IMAGE_FILE_ID_RE.findall(value))
            _add_unique(sediment_ids, SEDIMENT_ID_RE.findall(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return file_ids, sediment_ids


def _has_image_asset_pointer(payload: Any) -> bool:
    if isinstance(payload, dict):
        if str(payload.get("content_type") or "") == "image_asset_pointer":
            return True
        if str(payload.get("asset_pointer") or "").startswith(("file-service://", "sediment://")):
            return True
        return any(_has_image_asset_pointer(item) for item in payload.values())
    if isinstance(payload, list):
        return any(_has_image_asset_pointer(item) for item in payload)
    return False


class OpenAIWebImageClient:
    """ChatGPT Web picture_v2 client for image generation and editing."""

    def __init__(
        self,
        access_token: str,
        *,
        proxy: str = "",
        upstream_model: str = "gpt-5-5",
        thinking_effort: str = "",
        session: Any = None,
        poll_timeout_secs: float = 150.0,
        initial_wait_secs: float = 8.0,
        poll_interval_secs: float = 6.0,
        settle_secs: float = 1.5,
        cleanup_conversation: bool = True,
    ) -> None:
        if not access_token:
            raise ImageWorkerError("Missing access_token", http_status=401, error_code="unauthorized")
        self.base_url = "https://chatgpt.com"
        self.access_token = access_token
        self.upstream_model = str(upstream_model or "gpt-5-5").strip() or "gpt-5-5"
        self.thinking_effort = str(thinking_effort or "").strip().lower()
        if self.thinking_effort in {"none", "auto"}:
            self.thinking_effort = ""
        self.user_agent = DEFAULT_USER_AGENT
        self.device_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.pow_script_sources: list[str] = []
        self.pow_data_build = ""
        self.poll_timeout_secs = max(0.05, float(poll_timeout_secs))
        self.initial_wait_secs = max(0.0, float(initial_wait_secs))
        self.poll_interval_secs = max(0.05, float(poll_interval_secs))
        self.settle_secs = max(0.0, float(settle_secs))
        self.cleanup_conversation = bool(cleanup_conversation)

        if session is not None:
            self.session = session
            self._owns_session = False
            self.proxy_url = ""
        else:
            if requests is None:
                raise ImageWorkerError(
                    "curl_cffi is not installed in worker Python environment",
                    http_status=500,
                    error_code="dependencies_missing",
                )
            self.proxy_url = resolve_proxy(proxy)
            session_kwargs: Dict[str, Any] = {"impersonate": "chrome110", "verify": True}
            if self.proxy_url:
                session_kwargs["proxy"] = self.proxy_url
            self.session = requests.Session(**session_kwargs)
            self._owns_session = True

        self._base_headers = {
            "User-Agent": self.user_agent,
            "Origin": self.base_url,
            "Referer": self.base_url + "/",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Sec-Ch-Ua": DEFAULT_SEC_CH_UA,
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "OAI-Device-Id": self.device_id,
            "OAI-Session-Id": self.session_id,
            "OAI-Language": "zh-CN",
            "OAI-Client-Version": DEFAULT_CLIENT_VERSION,
            "OAI-Client-Build-Number": DEFAULT_CLIENT_BUILD_NUMBER,
            "Authorization": f"Bearer {self.access_token}",
        }

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self._owns_session and hasattr(self.session, "close"):
            try:
                self.session.close()
            except Exception:
                pass

    def __enter__(self) -> "OpenAIWebImageClient":
        return self

    def __exit__(self, *_: Any) -> bool:
        self.close()
        return False

    def _headers(self, path: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = dict(self._base_headers)
        headers["X-OpenAI-Target-Path"] = path
        headers["X-OpenAI-Target-Route"] = path
        if extra:
            headers.update(extra)
        return headers

    def _bootstrap_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Sec-Ch-Ua": DEFAULT_SEC_CH_UA,
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

    def _image_headers(
        self,
        path: str,
        requirements: ChatRequirements,
        conduit_token: str = "",
        accept: str = "*/*",
    ) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "OpenAI-Sentinel-Chat-Requirements-Token": requirements.token,
        }
        if requirements.proof_token:
            headers["OpenAI-Sentinel-Proof-Token"] = requirements.proof_token
        if conduit_token:
            headers["X-Conduit-Token"] = conduit_token
        if accept == "text/event-stream":
            headers["X-Oai-Turn-Trace-Id"] = str(uuid.uuid4())
        return self._headers(path, headers)

    @staticmethod
    def _extract_retry_after_from_body(body: Any) -> int | None:
        if not isinstance(body, dict):
            return None
        candidates = [body.get("retry_after"), body.get("resets_in_seconds"), body.get("clears_in")]
        for nested_key in ("detail", "error"):
            nested = body.get(nested_key)
            if isinstance(nested, dict):
                candidates.extend([
                    nested.get("retry_after"),
                    nested.get("resets_in_seconds"),
                    nested.get("clears_in"),
                ])
        for candidate in candidates:
            if candidate is None or isinstance(candidate, bool):
                continue
            try:
                sec = int(float(candidate))
            except (TypeError, ValueError):
                continue
            if sec > 0:
                return sec
        return None

    @staticmethod
    def _ensure_ok(response: Any, context: str) -> None:
        status = int(getattr(response, "status_code", 0) or 0)
        if 200 <= status < 300:
            return
        body: Any = getattr(response, "text", "")
        try:
            body = response.json()
        except Exception:
            pass
        retry_after: int | None = None
        headers = getattr(response, "headers", None)
        if headers:
            ra_raw = headers.get("Retry-After") if hasattr(headers, "get") else None
            if ra_raw is not None and str(ra_raw).strip().isdigit():
                retry_after = int(str(ra_raw).strip())
        if retry_after is None:
            retry_after = OpenAIWebImageClient._extract_retry_after_from_body(body)
        if _is_token_invalid_text(json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)):
            raise UpstreamHTTPError(context, 401, body, retry_after=retry_after)
        raise UpstreamHTTPError(context, status, body, retry_after=retry_after)

    @staticmethod
    def extract_image_quota(limits_progress: Any) -> tuple[int, str | None, bool]:
        if isinstance(limits_progress, list):
            for item in limits_progress:
                if isinstance(item, dict) and item.get("feature_name") == "image_gen":
                    raw_remaining = item.get("remaining")
                    try:
                        quota = max(0, int(raw_remaining if raw_remaining is not None else 0))
                    except (TypeError, ValueError):
                        quota = 0
                    restore_at = str(item.get("reset_after") or "").strip() or None
                    return quota, restore_at, True
        return 0, None, False

    def check_token(self, *, timezone_offset_min: int = -480) -> Dict[str, Any]:
        """Lightweight Web-side token probe used to admit Platform OAuth tokens."""
        chk_path = "/backend-api/accounts/check/v4-2023-04-27"
        try:
            chk_resp = self.session.get(
                f"{self.base_url}{chk_path}?timezone_offset_min={int(timezone_offset_min)}",
                headers=self._headers(chk_path, {"Accept": "application/json"}),
                timeout=20,
            )
        except Exception as exc:
            raise ImageWorkerError(
                "accounts_check_network_error",
                http_status=502,
                error_code="network_error",
            ) from exc

        status = int(getattr(chk_resp, "status_code", 0) or 0)
        if status == 401:
            return {"ok": False, "http_status": 401, "error_code": "unauthorized", "error": "Access token invalidated"}
        if status == 403:
            return {
                "ok": False,
                "http_status": 403,
                "error_code": "forbidden",
                "error": "Cloudflare / web endpoint denied the request",
            }
        self._ensure_ok(chk_resp, "accounts_check")
        plan_type = ""
        try:
            chk_data = chk_resp.json()
            if isinstance(chk_data, dict):
                default_acc = ((chk_data.get("accounts") or {}).get("default") or {}).get("account") or {}
                if isinstance(default_acc, dict):
                    plan_type = str(default_acc.get("plan_type") or "").strip()
        except Exception:
            pass
        return {"ok": True, "http_status": 200, "plan_type": plan_type or None}

    def check_image_quota(self, *, timezone_offset_min: int = -480) -> Dict[str, Any]:
        init_path = "/backend-api/conversation/init"
        init_resp = self.session.post(
            self.base_url + init_path,
            headers=self._headers(init_path, {"Content-Type": "application/json", "Accept": "application/json"}),
            json={
                "gizmo_id": None,
                "requested_default_model": None,
                "conversation_id": None,
                "timezone_offset_min": int(timezone_offset_min),
            },
            timeout=20,
        )
        self._ensure_ok(init_resp, "conversation_init")
        try:
            init_payload = init_resp.json()
        except Exception as exc:
            raise ImageWorkerError(
                "Invalid conversation init response",
                http_status=502,
                error_code="upstream_error",
            ) from exc
        if not isinstance(init_payload, dict):
            raise ImageWorkerError(
                "Invalid conversation init response",
                http_status=502,
                error_code="upstream_error",
            )

        quota, restore_at, has_image_limit = self.extract_image_quota(init_payload.get("limits_progress"))
        default_model_slug = str(init_payload.get("default_model_slug") or "").strip() or None

        plan_type = ""
        try:
            chk_path = "/backend-api/accounts/check/v4-2023-04-27"
            chk_resp = self.session.get(
                f"{self.base_url}{chk_path}?timezone_offset_min={int(timezone_offset_min)}",
                headers=self._headers(chk_path, {"Accept": "application/json"}),
                timeout=12,
            )
            if int(getattr(chk_resp, "status_code", 0) or 0) == 401:
                self._ensure_ok(chk_resp, "accounts_check")
            if int(getattr(chk_resp, "status_code", 0) or 0) == 200:
                chk_data = chk_resp.json()
                if isinstance(chk_data, dict):
                    default_acc = ((chk_data.get("accounts") or {}).get("default") or {}).get("account") or {}
                    if isinstance(default_acc, dict):
                        plan_type = str(default_acc.get("plan_type") or "").strip()
        except UpstreamHTTPError:
            raise
        except Exception:
            pass

        return {
            "ok": True,
            "quota": quota,
            "restore_at": restore_at,
            "has_image_limit": has_image_limit,
            "plan_type": plan_type or None,
            "default_model_slug": default_model_slug,
            "checked_at": int(time.time()),
        }

    def _bootstrap(self) -> None:
        response = self.session.get(
            self.base_url + "/",
            headers=self._bootstrap_headers(),
            timeout=30,
        )
        self._ensure_ok(response, "bootstrap")
        self.pow_script_sources, self.pow_data_build = parse_pow_resources(getattr(response, "text", "") or "")
        if not self.pow_script_sources:
            self.pow_script_sources = [DEFAULT_POW_SCRIPT]

    def _get_chat_requirements(self) -> ChatRequirements:
        base = "/backend-api/sentinel/chat-requirements"
        p_token = build_legacy_requirements_token(self.user_agent, self.pow_script_sources, self.pow_data_build)

        prepare_path = base + "/prepare"
        response = self.session.post(
            self.base_url + prepare_path,
            headers=self._headers(prepare_path, {"Content-Type": "application/json"}),
            json={"p": p_token},
            timeout=30,
        )
        self._ensure_ok(response, "chat_requirements_prepare")
        prepare_data = response.json()

        if (prepare_data.get("arkose") or {}).get("required"):
            raise ImageWorkerError(
                "Account requires Arkose challenge which is not supported",
                http_status=503,
                error_code="arkose_required",
            )

        proof_token = ""
        proof_info = prepare_data.get("proofofwork") or {}
        if proof_info.get("required"):
            proof_token = build_proof_token(
                str(proof_info.get("seed") or ""),
                str(proof_info.get("difficulty") or ""),
                self.user_agent,
                script_sources=self.pow_script_sources,
                data_build=self.pow_data_build,
            )

        turnstile_token = ""
        turnstile_info = prepare_data.get("turnstile") or {}
        if turnstile_info.get("required") and turnstile_info.get("dx"):
            turnstile_token = solve_turnstile_token(turnstile_info["dx"], p_token) or ""

        finalize_path = base + "/finalize"
        response = self.session.post(
            self.base_url + finalize_path,
            headers=self._headers(finalize_path, {"Content-Type": "application/json"}),
            json={
                "prepare_token": prepare_data.get("prepare_token", ""),
                "proof_token": proof_token,
                "turnstile_token": turnstile_token,
            },
            timeout=30,
        )
        self._ensure_ok(response, "chat_requirements_finalize")
        data = response.json()
        token = str(data.get("token") or "").strip()
        if not token:
            raise ImageWorkerError("Missing sentinel chat requirements token", http_status=502, error_code="upstream_error")

        return ChatRequirements(
            token=token,
            proof_token=proof_token,
            turnstile_token=turnstile_token,
            so_token=str(data.get("so_token") or ""),
            raw_finalize=data,
        )

    def _upload_image(self, data: bytes, file_name: str, mime_hint: str = "") -> Dict[str, Any]:
        _, mime_type, width, height = detect_image_info(data)
        if not mime_type:
            mime_type = mime_hint or "image/png"
        path = "/backend-api/files"
        response = self.session.post(
            self.base_url + path,
            headers=self._headers(path, {"Content-Type": "application/json", "Accept": "application/json"}),
            json={
                "file_name": file_name,
                "file_size": len(data),
                "use_case": "multimodal",
                "width": width,
                "height": height,
            },
            timeout=60,
        )
        self._ensure_ok(response, path)
        upload_meta = response.json()
        upload_url = _validate_https_url(upload_meta.get("upload_url", ""), "upload_url")
        file_id = str(upload_meta.get("file_id") or "").strip()
        if not file_id:
            raise ImageWorkerError("Upstream /files did not return file_id", http_status=502, error_code="upstream_error")

        put_resp = self.session.put(
            upload_url,
            headers={
                "Content-Type": mime_type,
                "x-ms-blob-type": "BlockBlob",
                "x-ms-version": "2020-04-08",
                "Origin": self.base_url,
                "Referer": self.base_url + "/",
                "User-Agent": self.user_agent,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.8",
            },
            data=data,
            timeout=120,
        )
        self._ensure_ok(put_resp, "image_upload")

        done_path = f"/backend-api/files/{file_id}/uploaded"
        done_resp = self.session.post(
            self.base_url + done_path,
            headers=self._headers(done_path, {"Content-Type": "application/json", "Accept": "application/json"}),
            data="{}",
            timeout=60,
        )
        self._ensure_ok(done_resp, done_path)
        return {
            "file_id": file_id,
            "file_name": file_name,
            "file_size": len(data),
            "mime_type": mime_type,
            "width": width,
            "height": height,
        }

    def _prepare_image_conversation(self, prompt: str, requirements: ChatRequirements) -> str:
        path = "/backend-api/f/conversation/prepare"
        payload: Dict[str, Any] = {
            "action": "next",
            "fork_from_shared_post": False,
            "parent_message_id": str(uuid.uuid4()),
            "model": self.upstream_model,
            "client_prepare_state": "success",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "system_hints": ["picture_v2"],
            "partial_query": {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
            },
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {"app_name": "chatgpt.com"},
        }
        if self.thinking_effort:
            payload["thinking_effort"] = self.thinking_effort
        response = self.session.post(
            self.base_url + path,
            headers=self._image_headers(path, requirements),
            json=payload,
            timeout=60,
        )
        self._ensure_ok(response, path)
        return str((response.json() or {}).get("conduit_token") or "")

    def _start_image_generation(
        self,
        prompt: str,
        requirements: ChatRequirements,
        conduit_token: str,
        references: list[Dict[str, Any]],
    ) -> Any:
        parts: list[Any] = [
            {
                "content_type": "image_asset_pointer",
                "asset_pointer": f"file-service://{item['file_id']}",
                "width": item["width"],
                "height": item["height"],
                "size_bytes": item["file_size"],
            }
            for item in references
        ]
        parts.append(prompt)
        content = (
            {"content_type": "multimodal_text", "parts": parts}
            if references
            else {"content_type": "text", "parts": [prompt]}
        )
        metadata: Dict[str, Any] = {
            "developer_mode_connector_ids": [],
            "selected_github_repos": [],
            "selected_all_github_repos": False,
            "system_hints": ["picture_v2"],
            "serialization_metadata": {"custom_symbol_offsets": []},
        }
        if references:
            metadata["attachments"] = [
                {
                    "id": item["file_id"],
                    "mimeType": item["mime_type"],
                    "name": item["file_name"],
                    "size": item["file_size"],
                    "width": item["width"],
                    "height": item["height"],
                }
                for item in references
            ]
        payload: Dict[str, Any] = {
            "action": "next",
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "author": {"role": "user"},
                    "create_time": time.time(),
                    "content": content,
                    "metadata": metadata,
                }
            ],
            "parent_message_id": str(uuid.uuid4()),
            "model": self.upstream_model,
            "client_prepare_state": "sent",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": True,
            "system_hints": ["picture_v2"],
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "is_dark_mode": False,
                "time_since_loaded": 1200,
                "page_height": 1072,
                "page_width": 1724,
                "pixel_ratio": 1.2,
                "screen_height": 1440,
                "screen_width": 2560,
                "app_name": "chatgpt.com",
            },
            "paragen_cot_summary_display_override": "allow",
            "force_parallel_switch": "auto",
        }
        if self.thinking_effort:
            payload["thinking_effort"] = self.thinking_effort

        path = "/backend-api/f/conversation"
        response = self.session.post(
            self.base_url + path,
            headers=self._image_headers(path, requirements, conduit_token, "text/event-stream"),
            json=payload,
            timeout=300,
            stream=True,
        )
        self._ensure_ok(response, path)
        return response

    def _consume_image_sse(self, response: Any, hard_cap_secs: float) -> ConversationState:
        state = ConversationState()
        deadline = time.monotonic() + hard_cap_secs
        close_fn = getattr(response, "close", None)
        watchdog = threading.Timer(hard_cap_secs, close_fn) if callable(close_fn) else None
        if watchdog is not None:
            watchdog.daemon = True
            watchdog.start()
        timeout_msg = f"Image generation stream exceeded hard timeout of {int(hard_cap_secs)}s"
        try:
            for payload in _iter_sse_payloads(response):
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    _update_conversation_state(state, payload, None)
                    continue
                if not isinstance(event, dict):
                    continue
                _update_conversation_state(state, payload, event)
                next_raw = _assistant_raw_text(event, state.raw_text)
                state.raw_text = next_raw
                state.text = _sanitize_output_text(next_raw)
                if time.monotonic() >= deadline:
                    raise ImageStreamHardTimeoutError(timeout_msg, state.conversation_id)
        except ImageStreamHardTimeoutError:
            raise
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise ImageStreamHardTimeoutError(timeout_msg, state.conversation_id) from exc
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        return state

    def _get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        path = f"/backend-api/conversation/{conversation_id}"
        response = self.session.get(
            self.base_url + path,
            headers=self._headers(path, {"Accept": "application/json"}),
            timeout=60,
        )
        self._ensure_ok(response, path)
        return response.json()

    def _list_recent_conversations(self, limit: int = 10, timeout_secs: float = 10.0) -> list[Dict[str, Any]]:
        path = f"/backend-api/conversations?offset=0&limit={limit}&order=updated&conversation_filter=all"
        try:
            response = self.session.get(
                self.base_url + path,
                headers=self._headers(path, {"Accept": "application/json"}),
                timeout=timeout_secs,
            )
            self._ensure_ok(response, path)
            data = response.json()
            return data.get("items") or data.get("conversations") or []
        except Exception:
            return []

    def find_conversation_by_prompt(self, prompt: str, started_at: float, timeout_secs: float = 10.0) -> str:
        items = self._list_recent_conversations(limit=10, timeout_secs=timeout_secs)
        if not items:
            return ""
        prompt_lower = str(prompt or "").lower().strip()
        best_match = ""
        best_score = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            conv_id = str(item.get("id") or item.get("conversation_id") or "")
            if not conv_id:
                continue
            updated_at = float(item.get("update_time") or item.get("updated_at") or 0)
            if updated_at and started_at and (updated_at < started_at - 30 or updated_at > started_at + 600):
                continue
            title = str(item.get("title") or "").lower()
            score = 0.0
            if prompt_lower and title:
                p_words = set(prompt_lower.split())
                t_words = set(title.split())
                common = p_words & t_words
                if common:
                    score = len(common) / max(len(p_words), 1)
            if title.startswith("image"):
                score += 0.3
            if score > best_score:
                best_score = score
                best_match = conv_id
        if best_match and best_score > 0.1:
            return best_match
        for item in items:
            if not isinstance(item, dict):
                continue
            conv_id = str(item.get("id") or item.get("conversation_id") or "")
            updated_at = float(item.get("update_time") or item.get("updated_at") or 0)
            if conv_id and updated_at and started_at and updated_at >= started_at - 30:
                return conv_id
        return ""

    def _query_backend_tasks(self, conversation_id: str = "", timeout_secs: float = 10.0) -> list[Dict[str, Any]]:
        path = "/backend-api/tasks"
        response = self.session.get(
            self.base_url + path,
            headers=self._headers(path, {"Accept": "application/json"}),
            timeout=timeout_secs,
        )
        self._ensure_ok(response, path)
        tasks = (response.json() or {}).get("tasks", [])
        if not isinstance(tasks, list):
            return []
        if conversation_id:
            tasks = [
                t
                for t in tasks
                if isinstance(t, dict)
                and (t.get("conversation_id") == conversation_id or t.get("original_conversation_id") == conversation_id)
            ]
        return tasks

    @staticmethod
    def check_task_error(task: Dict[str, Any]) -> tuple[bool, str]:
        img_msg = task.get("image_gen_message") or {}
        if not isinstance(img_msg, dict) or not img_msg:
            return False, ""
        metadata = img_msg.get("metadata") or {}
        content = img_msg.get("content") or {}
        is_error = bool(metadata.get("is_error", False))
        is_text_only = content.get("content_type") == "text"
        error_msg = ""
        if is_error and is_text_only:
            parts = content.get("parts") or []
            if isinstance(parts, list):
                error_msg = "".join(p for p in parts if isinstance(p, str))
        return is_error, error_msg

    def _get_detailed_task_error(self, conversation_id: str, wait_secs: float = 0.0) -> str:
        if not conversation_id:
            return ""
        try:
            if wait_secs > 0:
                time.sleep(wait_secs)
            for task in self._query_backend_tasks(conversation_id=conversation_id, timeout_secs=5.0):
                is_error, error_msg = self.check_task_error(task)
                if is_error and error_msg:
                    return error_msg
        except Exception:
            pass
        return ""

    @staticmethod
    def _extract_image_tool_records(data: Dict[str, Any]) -> list[Dict[str, Any]]:
        mapping = data.get("mapping") or {}
        records: list[Dict[str, Any]] = []
        for message_id, node in mapping.items():
            message = (node or {}).get("message") or {}
            author = message.get("author") or {}
            metadata = message.get("metadata") or {}
            content = message.get("content") or {}
            role = str(author.get("role") or "").strip().lower()
            if role not in {"tool", "assistant"}:
                continue
            is_image_gen = metadata.get("async_task_type") == "image_gen"
            has_pointer = _has_image_asset_pointer(content) or _has_image_asset_pointer(metadata)
            if role == "assistant" and not (is_image_gen or has_pointer):
                continue
            file_ids, sediment_ids = _extract_image_reference_ids({"content": content, "metadata": metadata})
            if not is_image_gen and not has_pointer and not file_ids and not sediment_ids:
                continue
            records.append(
                {
                    "message_id": message_id,
                    "create_time": message.get("create_time") or 0,
                    "file_ids": file_ids,
                    "sediment_ids": sediment_ids,
                }
            )
        return sorted(records, key=lambda item: float(item["create_time"] or 0))

    @staticmethod
    def _find_content_policy_error_in_conversation(data: Dict[str, Any]) -> str:
        mapping = data.get("mapping") or {}
        for node in mapping.values():
            message = (node or {}).get("message") or {}
            role = str(((message.get("author") or {}).get("role")) or "").strip().lower()
            if role not in {"assistant", "tool"}:
                continue
            content = message.get("content") or {}
            text_parts: list[str] = []
            if isinstance(content, dict):
                for part in content.get("parts") or []:
                    if isinstance(part, str) and part.strip():
                        text_parts.append(part.strip())
                text_field = str(content.get("text") or "").strip()
                if text_field:
                    text_parts.append(text_field)
            elif isinstance(content, str) and content.strip():
                text_parts.append(content.strip())
            msg_text = "\n".join(text_parts)
            if msg_text and _is_content_policy_error(msg_text):
                return msg_text[:500]
        return ""

    def _poll_image_results(
        self,
        conversation_id: str,
        initial_file_ids: list[str],
        initial_sediment_ids: list[str],
        timeout_secs: float,
    ) -> tuple[list[str], list[str]]:
        start = time.monotonic()
        attempt = 0
        file_ids = list(dict.fromkeys(x for x in initial_file_ids if x and x != "file_upload"))
        sediment_ids = list(dict.fromkeys(x for x in initial_sediment_ids if x))
        has_initial_ids = bool(file_ids or sediment_ids)
        last_hit_key: tuple[tuple[str, ...], tuple[str, ...]] | None = (
            (tuple(file_ids), tuple(sediment_ids)) if has_initial_ids else None
        )

        def _remaining() -> float:
            return timeout_secs - (time.monotonic() - start)

        if has_initial_ids:
            if self.settle_secs > 0:
                settle_for = min(self.settle_secs, max(0.0, _remaining()))
                if settle_for > 0:
                    time.sleep(settle_for)
        elif self.initial_wait_secs > 0:
            jitter = random.uniform(0, min(1.5, self.initial_wait_secs * 0.2))
            sleep_for = min(self.initial_wait_secs + jitter, max(0.0, _remaining()))
            if sleep_for > 0:
                time.sleep(sleep_for)

        last_task_error = ""
        while _remaining() > 0:
            attempt += 1
            last_task_error = self._get_detailed_task_error(conversation_id, wait_secs=0.0)
            try:
                conversation = self._get_conversation(conversation_id)
            except UpstreamHTTPError as exc:
                if exc.status_code in (429, 500, 502, 503, 504):
                    base = exc.retry_after if exc.retry_after is not None else min(2 ** min(attempt, 4), 16)
                    sleep_for = min(base + random.uniform(0, 0.3), max(0.0, _remaining()))
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                        continue
                    break
                raise
            except Exception:
                sleep_for = min(min(2 ** min(attempt, 3), 8), max(0.0, _remaining()))
                if sleep_for > 0:
                    time.sleep(sleep_for)
                    continue
                break

            for record in self._extract_image_tool_records(conversation):
                _add_unique(file_ids, [x for x in record["file_ids"] if x != "file_upload"])
                _add_unique(sediment_ids, record["sediment_ids"])

            if not file_ids and not sediment_ids:
                policy_msg = self._find_content_policy_error_in_conversation(conversation)
                if policy_msg:
                    raise ImageContentPolicyError(policy_msg, conversation_id)

            if file_ids or sediment_ids:
                hit_key = (tuple(file_ids), tuple(sediment_ids))
                if self.settle_secs <= 0 or last_hit_key == hit_key:
                    return file_ids, sediment_ids
                last_hit_key = hit_key
                wait = min(self.settle_secs, max(0.0, _remaining()))
                if wait > 0:
                    time.sleep(wait)
                    continue
                return file_ids, sediment_ids

            wait = min(self.poll_interval_secs, max(0.0, _remaining()))
            if wait > 0:
                time.sleep(wait)

        if last_task_error and _is_content_policy_error(last_task_error):
            raise ImageContentPolicyError(last_task_error, conversation_id)
        raise ImagePollTimeoutError(
            f"ChatGPT image polling timed out after {int(timeout_secs)}s",
            conversation_id=conversation_id,
            task_error=last_task_error,
        )

    def _resolve_image_urls(self, conversation_id: str, file_ids: list[str], sediment_ids: list[str]) -> list[str]:
        urls: list[str] = []
        resolved_ids: set[str] = set()
        for file_id in file_ids:
            if not file_id or file_id == "file_upload":
                continue
            for path in (f"/backend-api/files/{file_id}/download", f"/backend-api/files/download/{file_id}"):
                try:
                    resp = self.session.get(
                        self.base_url + path,
                        headers=self._headers(path, {"Accept": "application/json"}),
                        timeout=60,
                    )
                    if 200 <= int(getattr(resp, "status_code", 0) or 0) < 300:
                        data = resp.json() or {}
                        url = str(data.get("download_url") or data.get("url") or "").strip()
                        if url:
                            _validate_https_url(url, "download_url")
                            resolved_ids.add(file_id)
                            if url not in urls:
                                urls.append(url)
                            break
                    elif int(getattr(resp, "status_code", 0) or 0) == 401:
                        self._ensure_ok(resp, path)
                except UpstreamHTTPError:
                    raise
                except Exception:
                    continue

        if conversation_id and sediment_ids:
            for sediment_id in sediment_ids:
                if not sediment_id or sediment_id in resolved_ids:
                    continue
                path = f"/backend-api/conversation/{conversation_id}/attachment/{sediment_id}/download"
                try:
                    resp = self.session.get(
                        self.base_url + path,
                        headers=self._headers(path, {"Accept": "application/json"}),
                        timeout=60,
                    )
                    if 200 <= int(getattr(resp, "status_code", 0) or 0) < 300:
                        data = resp.json() or {}
                        url = str(data.get("download_url") or data.get("url") or "").strip()
                        if url:
                            _validate_https_url(url, "download_url")
                            resolved_ids.add(sediment_id)
                            if url not in urls:
                                urls.append(url)
                    elif int(getattr(resp, "status_code", 0) or 0) == 401:
                        self._ensure_ok(resp, path)
                except UpstreamHTTPError:
                    raise
                except Exception:
                    continue
        return urls

    def _download_image_bytes(self, urls: list[str]) -> list[bytes]:
        images: list[bytes] = []
        seen_hashes: set[str] = set()
        for url in urls:
            safe_url = _validate_https_url(url, "download_url")
            host = (urlparse(safe_url).hostname or "").lower()
            is_chatgpt_host = host == "chatgpt.com" or host.endswith(".chatgpt.com")
            dl_headers: Dict[str, str] = {
                "User-Agent": self.user_agent,
                "Accept": "image/avif,image/webp,image/png,*/*",
            }
            if is_chatgpt_host:
                # /backend-api/estuary/content returns 404 when polluted with Origin/OAI-*/Sec-Fetch-Mode:cors;
                # send only clean Bearer + Referer on chatgpt.com and never leak Bearer to external CDN hosts.
                dl_headers["Authorization"] = f"Bearer {self.access_token}"
                dl_headers["Referer"] = self.base_url + "/"

            if self._owns_session and requests is not None:
                dl_kwargs: Dict[str, Any] = {"impersonate": "chrome110", "verify": True}
                if getattr(self, "proxy_url", ""):
                    dl_kwargs["proxy"] = self.proxy_url
                dl_session = requests.Session(**dl_kwargs)
                try:
                    if is_chatgpt_host and getattr(self.session, "cookies", None):
                        try:
                            dl_session.cookies.update(self.session.cookies)
                        except Exception:
                            pass
                    resp = dl_session.get(safe_url, headers=dl_headers, timeout=120)
                finally:
                    dl_session.close()
            else:
                resp = self.session.get(safe_url, headers=dl_headers, timeout=120)
            self._ensure_ok(resp, "image_download")
            content = bytes(getattr(resp, "content", b"") or b"")
            if not content:
                continue
            digest = hashlib.sha256(content).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                images.append(content)
        return images

    def delete_conversation(self, conversation_id: str) -> None:
        if not conversation_id:
            return
        path = f"/backend-api/conversation/{conversation_id}"
        headers = self._headers(
            path,
            {
                "Accept": "*/*",
                "Content-Type": "application/json",
                "Referer": f"{self.base_url}/c/{conversation_id}",
                "X-OpenAI-Target-Route": "/backend-api/conversation/{conversation_id}",
            },
        )
        self.session.patch(
            self.base_url + path,
            headers=headers,
            json={"is_visible": False},
            timeout=5,
        )

    def generate_image(
        self,
        *,
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        quality: str = "auto",
        output_format: str | None = None,
        background: str | None = None,
        frame_count: int | None = None,
        frame_duration_ms: int | None = None,
        images: list[tuple[bytes, str, str]] | None = None,
        masks: list[tuple[bytes, str, str]] | None = None,
    ) -> Dict[str, Any]:
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise ImageWorkerError("Prompt is required", http_status=400, error_code="invalid_request")
        base_model = normalize_image_model(model)
        if base_model is None:
            raise ImageWorkerError(
                f"Unsupported image model: {model}",
                http_status=400,
                error_code="model_not_found",
            )
        fmt_clean = str(output_format or "").strip().lower()
        if fmt_clean == "jpg":
            fmt_clean = "jpeg"
        if fmt_clean not in SUPPORTED_OUTPUT_FORMATS:
            raise ImageWorkerError(
                f"Unsupported output_format: {output_format}",
                http_status=400,
                error_code="invalid_request",
            )
        if fmt_clean == "gif" and base_model not in GIF_CAPABLE_MODELS:
            raise ImageWorkerError(
                "GIF 动图需要 gpt-image-2.6 系列模型",
                http_status=400,
                error_code="invalid_request",
            )
        self.upstream_model, self.thinking_effort = image_model_settings(base_model)
        output_format = fmt_clean or None

        input_images = list(images or [])
        input_masks = list(masks or [])
        if input_masks:
            if not input_images:
                raise ImageWorkerError("Mask requires at least one input image", http_status=400, error_code="invalid_request")
            input_images = composite_images_with_masks(input_images, input_masks)

        conversation_id = ""
        try:
            references = [
                self._upload_image(data, fname, mime_type)
                for data, fname, mime_type in input_images
            ]
            final_prompt = build_image_prompt(
                clean_prompt,
                size=size,
                quality=quality,
                output_format=output_format,
                background=background,
                frame_count=frame_count,
                frame_duration_ms=frame_duration_ms,
            )
            started_at = time.time()
            self._bootstrap()
            requirements = self._get_chat_requirements()
            conduit_token = self._prepare_image_conversation(final_prompt, requirements)
            sse_resp = self._start_image_generation(final_prompt, requirements, conduit_token, references)
            state = self._consume_image_sse(sse_resp, self.poll_timeout_secs)

            conversation_id = state.conversation_id
            file_ids = [x for x in state.file_ids if x != "file_upload"]
            sediment_ids = list(state.sediment_ids)
            message = state.text.strip()

            if message and not file_ids and not sediment_ids and state.blocked:
                detailed = self._get_detailed_task_error(conversation_id, wait_secs=min(1.0, self.settle_secs))
                raise ImageContentPolicyError(detailed or message, conversation_id)

            is_text_reply = bool(message and _is_model_text_reply_instead_of_image(message))
            should_poll = bool(references) or state.turn_use_case == "image gen" or is_text_reply

            if message and not file_ids and not sediment_ids and not should_poll:
                if _is_content_policy_error(message):
                    raise ImageContentPolicyError(message, conversation_id)
                raise ImageWorkerError(
                    _sanitize_error_text(message),
                    http_status=502,
                    error_code="upstream_text_reply",
                    conversation_id=conversation_id,
                )

            if should_poll and not conversation_id:
                conversation_id = self.find_conversation_by_prompt(final_prompt, started_at, timeout_secs=5.0)

            if not file_ids and not sediment_ids and conversation_id and not should_poll:
                detailed = self._get_detailed_task_error(conversation_id, wait_secs=min(0.5, self.settle_secs))
                if detailed:
                    if _is_content_policy_error(detailed):
                        raise ImageContentPolicyError(detailed, conversation_id)
                    raise ImageWorkerError(
                        _sanitize_error_text(detailed),
                        http_status=502,
                        error_code="upstream_error",
                        conversation_id=conversation_id,
                    )

            if conversation_id and ((not file_ids and not sediment_ids) or self.settle_secs > 0):
                polled_files, polled_sediments = self._poll_image_results(
                    conversation_id,
                    file_ids,
                    sediment_ids,
                    timeout_secs=self.poll_timeout_secs,
                )
                _add_unique(file_ids, polled_files)
                _add_unique(sediment_ids, polled_sediments)

            if not file_ids and not sediment_ids:
                raise ImageWorkerError(
                    "Upstream completed without returning image outputs",
                    http_status=502,
                    error_code="no_image_generated",
                    conversation_id=conversation_id,
                )

            urls = self._resolve_image_urls(conversation_id, file_ids, sediment_ids)
            if not urls:
                raise ImageWorkerError(
                    "Could not resolve downloadable image URLs from upstream",
                    http_status=502,
                    error_code="no_image_generated",
                    conversation_id=conversation_id,
                )

            downloaded = self._download_image_bytes(urls)
            if not downloaded:
                raise ImageWorkerError(
                    "Downloaded image payload was empty",
                    http_status=502,
                    error_code="no_image_generated",
                    conversation_id=conversation_id,
                )

            data_items = [
                {
                    "b64_json": base64.b64encode(raw_bytes).decode("ascii"),
                    "media_type": sniff_image_mime(raw_bytes),
                    "revised_prompt": clean_prompt,
                }
                for raw_bytes in downloaded
            ]
            return {
                "ok": True,
                "created": int(time.time()),
                "data": data_items,
            }
        finally:
            if self.cleanup_conversation and conversation_id:
                try:
                    self.delete_conversation(conversation_id)
                except Exception:
                    pass


def _decode_stdin_images(entries: Any, *, field_name: str = "images") -> list[tuple[bytes, str, str]]:
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ImageWorkerError(f"{field_name} must be a list", http_status=400, error_code="invalid_request")
    if len(entries) > MAX_INPUT_IMAGES:
        raise ImageWorkerError(
            f"{field_name} supports at most {MAX_INPUT_IMAGES} files",
            http_status=400,
            error_code="invalid_request",
        )
    decoded: list[tuple[bytes, str, str]] = []
    for idx, item in enumerate(entries, start=1):
        if not isinstance(item, dict):
            raise ImageWorkerError(f"{field_name}[{idx - 1}] must be an object", http_status=400, error_code="invalid_request")
        b64 = str(item.get("b64") or "").strip()
        if not b64:
            raise ImageWorkerError(f"{field_name}[{idx - 1}] is missing b64", http_status=400, error_code="invalid_image")
        try:
            raw = base64.b64decode(b64, validate=True)
        except Exception as exc:
            raise ImageWorkerError(f"Invalid base64 image in {field_name}", http_status=400, error_code="invalid_image") from exc
        if not raw:
            raise ImageWorkerError(f"Empty image in {field_name}", http_status=400, error_code="invalid_image")
        if len(raw) > MAX_INPUT_IMAGE_BYTES:
            raise ImageWorkerError("Input image exceeds 10MB limit", http_status=400, error_code="invalid_image")
        filename = Path(str(item.get("filename") or f"image_{idx}.png")).name.replace("\x00", "") or f"image_{idx}.png"
        mime = str(item.get("mime_type") or sniff_image_mime(raw)).strip().lower()
        decoded.append((raw, filename, mime))
    return decoded


def handle_payload(data: Any, *, session: Any = None) -> Dict[str, Any]:
    action = data.get("action") if isinstance(data, dict) else None
    if not isinstance(data, dict) or action not in ("generate_image", "check_image_quota", "check_token"):
        return {"ok": False, "http_status": 400, "error_code": "invalid_request", "error": "Invalid worker action"}

    access_token = str(data.get("access_token") or "").strip()
    if not access_token:
        return {"ok": False, "http_status": 401, "error_code": "unauthorized", "error": "Missing access_token"}

    try:
        if action == "check_token":
            tz_raw = data.get("timezone_offset_min")
            try:
                tz_offset = int(tz_raw) if tz_raw is not None else -480
            except (TypeError, ValueError):
                tz_offset = -480
            with OpenAIWebImageClient(
                access_token,
                proxy=str(data.get("proxy") or "").strip(),
                session=session,
            ) as client:
                return client.check_token(timezone_offset_min=tz_offset)

        if action == "check_image_quota":
            tz_raw = data.get("timezone_offset_min")
            try:
                tz_offset = int(tz_raw) if tz_raw is not None else -480
            except (TypeError, ValueError):
                tz_offset = -480
            with OpenAIWebImageClient(
                access_token,
                proxy=str(data.get("proxy") or "").strip(),
                session=session,
            ) as client:
                return client.check_image_quota(timezone_offset_min=tz_offset)

        clean_prompt = str(data.get("prompt") or "").strip()
        if not clean_prompt:
            raise ImageWorkerError("Prompt is required", http_status=400, error_code="invalid_request")
        requested_model = str(data.get("model") or data.get("model_id") or "").strip()
        base_model = normalize_image_model(requested_model)
        if base_model is None:
            raise ImageWorkerError(
                f"Unsupported image model: {requested_model}",
                http_status=400,
                error_code="model_not_found",
            )
        fmt_clean = str(data.get("output_format") or "").strip().lower()
        if fmt_clean == "jpg":
            fmt_clean = "jpeg"
        if fmt_clean not in SUPPORTED_OUTPUT_FORMATS:
            raise ImageWorkerError(
                f"Unsupported output_format: {data.get('output_format')}",
                http_status=400,
                error_code="invalid_request",
            )
        if fmt_clean == "gif" and base_model not in GIF_CAPABLE_MODELS:
            raise ImageWorkerError(
                "GIF 动图需要 gpt-image-2.6 系列模型",
                http_status=400,
                error_code="invalid_request",
            )
        frame_count_raw = data.get("frame_count")
        frame_duration_raw = data.get("frame_duration_ms")
        try:
            frame_count = int(frame_count_raw) if frame_count_raw not in (None, "") else None
        except (TypeError, ValueError):
            frame_count = None
        try:
            frame_duration_ms = (
                int(frame_duration_raw) if frame_duration_raw not in (None, "") else None
            )
        except (TypeError, ValueError):
            frame_duration_ms = None
        images = _decode_stdin_images(data.get("images"), field_name="images")
        masks = _decode_stdin_images(data.get("masks"), field_name="masks")
        if masks and not images:
            raise ImageWorkerError("Mask requires at least one input image", http_status=400, error_code="invalid_request")
        poll_timeout = max(0.05, min(300.0, float(data.get("poll_timeout_secs") or 150.0)))
        initial_wait = max(0.0, min(60.0, float(data.get("initial_wait_secs") if data.get("initial_wait_secs") is not None else 8.0)))
        poll_interval = max(0.05, min(30.0, float(data.get("poll_interval_secs") if data.get("poll_interval_secs") is not None else 6.0)))
        settle_secs = max(0.0, min(10.0, float(data.get("settle_secs") if data.get("settle_secs") is not None else 1.5)))
        cleanup = bool(data.get("cleanup_conversation", True))

        with OpenAIWebImageClient(
            access_token,
            proxy=str(data.get("proxy") or "").strip(),
            upstream_model=str(data.get("upstream_model") or DEFAULT_UPSTREAM_MODEL).strip(),
            thinking_effort=str(data.get("thinking_effort") or "").strip(),
            session=session,
            poll_timeout_secs=poll_timeout,
            initial_wait_secs=initial_wait,
            poll_interval_secs=poll_interval,
            settle_secs=settle_secs,
            cleanup_conversation=cleanup,
        ) as client:
            return client.generate_image(
                prompt=clean_prompt,
                model=base_model,
                size=str(data.get("size") or "") or None,
                quality=str(data.get("quality") or "auto"),
                output_format=fmt_clean or None,
                background=str(data.get("background") or "") or None,
                frame_count=frame_count,
                frame_duration_ms=frame_duration_ms,
                images=images,
                masks=masks,
            )
    except ImageWorkerError as exc:
        return {
            "ok": False,
            "http_status": exc.http_status,
            "error_code": exc.error_code,
            "error": _sanitize_error_text(str(exc)),
            **({"retry_after": exc.retry_after} if exc.retry_after is not None else {}),
        }
    except UpstreamHTTPError as exc:
        status = int(exc.status_code or 502)
        if status == 401:
            code = "unauthorized"
            out_status = 401
        elif status == 429:
            code = "rate_limited"
            out_status = 429
        elif status == 403:
            code = "forbidden"
            out_status = 403
        elif status in (400, 422):
            body_text = json.dumps(exc.body, ensure_ascii=False) if isinstance(exc.body, (dict, list)) else str(exc.body)
            if _is_content_policy_error(body_text):
                code = "content_policy_violation"
            else:
                code = "upstream_bad_request"
            out_status = 400
        elif status in (504,):
            code = "timeout"
            out_status = 504
        else:
            code = "upstream_error"
            out_status = 502
        return {
            "ok": False,
            "http_status": out_status,
            "error_code": code,
            "error": _sanitize_error_text(str(exc)),
            **({"retry_after": exc.retry_after} if exc.retry_after is not None else {}),
        }
    except Exception as exc:
        msg = str(exc)
        if _is_token_invalid_text(msg):
            return {
                "ok": False,
                "http_status": 401,
                "error_code": "unauthorized",
                "error": "Access token invalidated",
            }
        return {
            "ok": False,
            "http_status": 502,
            "error_code": "image_worker_error",
            "error": _sanitize_error_text(msg),
        }


def main() -> None:
    try:
        raw_bytes = sys.stdin.buffer.read(MAX_STDIN_BYTES)
        raw = raw_bytes.decode("utf-8") if raw_bytes else ""
        data = json.loads(raw) if raw else {}
        result = handle_payload(data)
        sys.stdout.buffer.write((json.dumps(result, ensure_ascii=True) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception as exc:
        err_payload = {
            "ok": False,
            "http_status": 500,
            "error_code": "image_worker_error",
            "error": _sanitize_error_text(str(exc)),
        }
        sys.stdout.buffer.write((json.dumps(err_payload, ensure_ascii=True) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
