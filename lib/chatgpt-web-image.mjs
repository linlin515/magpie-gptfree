/**
 * ChatGPT Web `picture_v2` image generation & editing adapter for Magpie GPTFree.
 *
 * Exposes `gpt-image-2` under Magpie's `@ai-sdk/openai-compatible` plugin model schema
 * (required by host.js / draw.go / TestModels) while keeping Codex text models on
 * `@ai-sdk/openai` (/responses). Dispatches isolated Python worker processes using
 * pooled ChatGPT access tokens with 401 reauthentication and multi-account failover.
 */
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { accessTokenState } from "./credentials.mjs";
import { workerPython } from "./dsh-http-auth.mjs";

const PYDIR = join(dirname(fileURLToPath(import.meta.url)), "../python");
export const IMAGE_WORKER = join(PYDIR, "image_worker.py");
export const IMAGE_MODEL_ID = "gpt-image-2";

/**
 * Image models supported through the ChatGPT Web `picture_v2` pipeline.
 * The 2.6 family is the newer generation also used for animated GIF output,
 * mirroring https://github.com/yang0/chatgpt2api-dsh.
 */
export const IMAGE_MODEL_IDS = Object.freeze([
  "gpt-image-2",
  "gpt-image-2.6",
  "gpt-image-2.6-flare",
  "gpt-image-2.6-sunburst",
]);

const IMAGE_MODEL_NAMES = {
  "gpt-image-2": "GPT Image 2",
  "gpt-image-2.6": "GPT Image 2.6",
  "gpt-image-2.6-flare": "GPT Image 2.6 Flare",
  "gpt-image-2.6-sunburst": "GPT Image 2.6 Sunburst",
};

const GIF_CAPABLE_MODELS = new Set([
  "gpt-image-2.6",
  "gpt-image-2.6-flare",
  "gpt-image-2.6-sunburst",
]);

const MAX_STDOUT_BYTES = 64 * 1024 * 1024;
const MAX_STDERR_BYTES = 256 * 1024;
const MAX_INPUT_FILES = 10;
const MAX_INPUT_FILE_BYTES = 10 * 1024 * 1024;
const ALLOWED_FORMATS = new Set(["", "auto", "png", "webp", "jpeg", "jpg", "gif"]);

export class ImageHttpError extends Error {
  constructor(status, message, code = "") {
    super(message);
    this.name = "ImageHttpError";
    this.status = status;
    this.code = code;
  }
}

function abortError(message = "Image generation aborted") {
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

export function normalizeImageModelId(modelId) {
  const raw = String(modelId || "").trim();
  if (!raw) return IMAGE_MODEL_ID;
  const lower = raw.toLowerCase();
  const hit = IMAGE_MODEL_IDS.find((id) => id === lower);
  if (hit) return hit;
  // Accept plan-prefixed aliases such as `plus-gpt-image-2.6`.
  const suffix = IMAGE_MODEL_IDS.find((id) => lower.endsWith("-" + id));
  return suffix || "";
}

export function isImageModelId(modelId) {
  // An explicitly empty model is not an image model (it defaults to a text model
  // request in the caller); only real ids or plan-prefixed aliases match.
  if (!String(modelId || "").trim()) return false;
  return normalizeImageModelId(modelId) !== "";
}

export function supportsGifOutput(modelId) {
  return GIF_CAPABLE_MODELS.has(normalizeImageModelId(modelId));
}

export function buildImageModelCatalog(baseURL = "https://chatgpt.com/backend-api/codex") {
  const catalog = {};
  for (const id of IMAGE_MODEL_IDS) {
    const gif = GIF_CAPABLE_MODELS.has(id);
    catalog[id] = {
      id,
      name: IMAGE_MODEL_NAMES[id],
      api: {
        id,
        url: baseURL,
        npm: "@ai-sdk/openai-compatible",
      },
      provider: {
        npm: "@ai-sdk/openai-compatible",
      },
      attachment: true,
      tool_call: false,
      temperature: false,
      limit: { context: 32000, output: 4096 },
      capabilities: {
        temperature: false,
        reasoning: false,
        attachment: true,
        toolcall: false,
        input: { text: true, image: true, audio: false, video: false, pdf: false },
        output: { text: false, image: true, audio: false, video: false, pdf: false },
        ...(gif ? { animated: true } : {}),
      },
    };
  }
  return catalog;
}

function headerValue(headers, targetName) {
  if (!headers) return "";
  const needle = targetName.toLowerCase();
  if (typeof headers.get === "function") {
    return String(headers.get(targetName) || headers.get(needle) || "").trim();
  }
  if (Array.isArray(headers)) {
    for (const pair of headers) {
      if (Array.isArray(pair) && String(pair[0] || "").toLowerCase() === needle) {
        return String(pair[1] || "").trim();
      }
    }
    return "";
  }
  if (typeof headers === "object") {
    for (const [k, v] of Object.entries(headers)) {
      if (k.toLowerCase() === needle) return String(v || "").trim();
    }
  }
  return "";
}

function parseN(raw) {
  if (raw === undefined || raw === null || raw === "") return 1;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1 || n > 4) {
    throw new ImageHttpError(400, "n must be an integer between 1 and 4", "invalid_request");
  }
  return n;
}

function normalizeOutputFormat(raw) {
  const fmt = String(raw || "").trim().toLowerCase();
  if (!ALLOWED_FORMATS.has(fmt)) {
    throw new ImageHttpError(400, "Unsupported image output_format", "invalid_request");
  }
  return fmt;
}

function parseFrameCount(raw) {
  if (raw === undefined || raw === null || String(raw).trim() === "") return undefined;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 2 || n > 120) {
    throw new ImageHttpError(400, "frame_count must be an integer between 2 and 120", "invalid_request");
  }
  return n;
}

function parseFrameDuration(raw) {
  if (raw === undefined || raw === null || String(raw).trim() === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 20 || n > 5000) {
    throw new ImageHttpError(400, "frame_duration_ms must be between 20 and 5000", "invalid_request");
  }
  return Math.trunc(n);
}

async function extractFormBlobs(form, keys) {
  const items = [];
  for (const key of keys) {
    for (const entry of form.getAll(key)) {
      if (!entry) continue;
      if (typeof entry !== "string" && typeof entry.arrayBuffer === "function") {
        const size = Number(entry.size || 0);
        if (size <= 0) {
          throw new ImageHttpError(400, `Uploaded file in "${key}" is empty`, "invalid_image");
        }
        if (size > MAX_INPUT_FILE_BYTES) {
          throw new ImageHttpError(400, `Uploaded file in "${key}" exceeds 10MB limit`, "invalid_image");
        }
        const buf = Buffer.from(await entry.arrayBuffer());
        if (!buf.length) {
          throw new ImageHttpError(400, `Uploaded file in "${key}" is empty`, "invalid_image");
        }
        items.push({
          b64: buf.toString("base64"),
          mime_type: String(entry.type || "").trim().toLowerCase(),
          filename: String(entry.name || "image.png").trim() || "image.png",
        });
      }
    }
  }
  return items;
}

function decodeJsonImageItem(entry, index) {
  let raw = "";
  let filename = `image_${index}.png`;
  let mimeType = "";
  if (typeof entry === "string") {
    raw = entry.trim();
  } else if (entry && typeof entry === "object") {
    filename = String(entry.filename || entry.file_name || filename).trim() || filename;
    mimeType = String(entry.mime_type || entry.mimeType || "").trim().toLowerCase();
    const candidate = entry.b64_json || entry.base64 || entry.image_url || entry.url;
    if (candidate && typeof candidate === "object") {
      raw = String(candidate.url || candidate.image_url || "").trim();
      mimeType = mimeType || String(candidate.mime_type || candidate.mimeType || "").trim().toLowerCase();
    } else if (typeof candidate === "string") {
      raw = candidate.trim();
    }
  } else {
    throw new ImageHttpError(400, "image entry must be a base64 string or object", "invalid_image");
  }

  if (!raw) throw new ImageHttpError(400, "image entry must include image data", "invalid_image");
  if (/^https?:\/\//i.test(raw)) {
    throw new ImageHttpError(400, "Remote image URLs are not supported; supply base64 or multipart file", "invalid_image");
  }

  const dataUrl = /^data:([^;]+);base64,([\s\S]*)$/i.exec(raw);
  let b64 = raw;
  if (dataUrl) {
    mimeType = mimeType || dataUrl[1].trim().toLowerCase();
    b64 = dataUrl[2].trim();
  }
  if (!b64) throw new ImageHttpError(400, "image file is empty", "invalid_image");

  let decoded;
  try {
    decoded = Buffer.from(b64, "base64");
  } catch {
    throw new ImageHttpError(400, "Invalid base64 image payload", "invalid_image");
  }
  if (!decoded.length) throw new ImageHttpError(400, "image file is empty", "invalid_image");
  if (decoded.length > MAX_INPUT_FILE_BYTES) {
    throw new ImageHttpError(400, "image file exceeds 10MB limit", "invalid_image");
  }
  return { b64: decoded.toString("base64"), mime_type: mimeType, filename };
}

function normalizeJsonImageList(rawValue, fieldName) {
  if (rawValue === undefined || rawValue === null) return [];
  const list = Array.isArray(rawValue) ? rawValue : [rawValue];
  if (list.length > MAX_INPUT_FILES) {
    throw new ImageHttpError(400, `${fieldName} supports at most ${MAX_INPUT_FILES} files`, "invalid_request");
  }
  return list.map((item, idx) => decodeJsonImageItem(item, idx + 1));
}

function parseJsonBody(rawBody) {
  if (rawBody && typeof rawBody === "object" && !(rawBody instanceof Uint8Array) && !ArrayBuffer.isView(rawBody) && !(rawBody instanceof ArrayBuffer)) {
    return rawBody;
  }
  const text = rawBody instanceof Uint8Array ? Buffer.from(rawBody).toString("utf8") :
    rawBody instanceof ArrayBuffer ? Buffer.from(rawBody).toString("utf8") :
    String(rawBody || "");
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new ImageHttpError(400, "Invalid JSON image request body", "invalid_request");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new ImageHttpError(400, "Image request body must be a JSON object", "invalid_request");
  }
  return parsed;
}

export async function parseImageRequest(pathname, rawBody, headers) {
  const isEdit = pathname.endsWith("/images/edits");
  const contentType = headerValue(headers, "content-type");

  if (isEdit && (rawBody instanceof FormData || /^multipart\/form-data\b/i.test(contentType))) {
    let form;
    if (rawBody instanceof FormData) {
      form = rawBody;
    } else {
      try {
        form = await new Response(rawBody, { headers: { "content-type": contentType } }).formData();
      } catch {
        throw new ImageHttpError(400, "Cannot parse multipart/form-data image edit request", "invalid_request");
      }
    }
    const textField = (k) => {
      const v = form.get(k);
      return typeof v === "string" ? v.trim() : "";
    };
    const prompt = textField("prompt");
    if (!prompt) throw new ImageHttpError(400, "prompt is required", "invalid_request");
    const requestedModel = textField("model");
    const model = normalizeImageModelId(requestedModel);
    if (!model) {
      throw new ImageHttpError(400, `Unsupported image model: ${requestedModel}`, "model_not_found");
    }
    const outputFormat = normalizeOutputFormat(textField("output_format"));
    if (outputFormat === "gif" && !supportsGifOutput(model)) {
      throw new ImageHttpError(400, "GIF 动图需要 gpt-image-2.6 系列模型", "invalid_request");
    }
    const images = await extractFormBlobs(form, ["image", "image[]"]);
    const markupImages = await extractFormBlobs(form, ["markup", "markup[]"]);
    const masks = await extractFormBlobs(form, ["mask", "mask[]"]);
    const combinedImages = [...images, ...markupImages];
    if (!combinedImages.length) {
      throw new ImageHttpError(400, "image file is required", "invalid_image");
    }
    if (combinedImages.length > MAX_INPUT_FILES) {
      throw new ImageHttpError(400, `images supports at most ${MAX_INPUT_FILES} files`, "invalid_request");
    }
    if (masks.length > MAX_INPUT_FILES) {
      throw new ImageHttpError(400, `masks supports at most ${MAX_INPUT_FILES} files`, "invalid_request");
    }
    return {
      model,
      prompt,
      n: parseN(textField("n")),
      size: textField("size"),
      quality: textField("quality") || "auto",
      background: textField("background") || "auto",
      output_format: outputFormat,
      frame_count: parseFrameCount(textField("frame_count")),
      frame_duration_ms: parseFrameDuration(textField("frame_duration_ms")),
      images: combinedImages,
      masks,
    };
  }

  const b = parseJsonBody(rawBody);
  const prompt = String(b.prompt || "").trim();
  if (!prompt) throw new ImageHttpError(400, "prompt is required", "invalid_request");
  const requestedModel = String(b.model || IMAGE_MODEL_ID).trim();
  const model = normalizeImageModelId(requestedModel);
  if (!model) {
    throw new ImageHttpError(400, `Unsupported image model: ${requestedModel}`, "model_not_found");
  }
  const outputFormat = normalizeOutputFormat(b.output_format);
  if (outputFormat === "gif" && !supportsGifOutput(model)) {
    throw new ImageHttpError(400, "GIF 动图需要 gpt-image-2.6 系列模型", "invalid_request");
  }

  let images = [];
  let masks = [];
  if (isEdit) {
    images = normalizeJsonImageList(b.images !== undefined ? b.images : b.image, "images");
    const markupImages = normalizeJsonImageList(b.markup, "markup");
    images = [...images, ...markupImages];
    if (!images.length) {
      throw new ImageHttpError(400, "image is required for edits", "invalid_image");
    }
    if (images.length > MAX_INPUT_FILES) {
      throw new ImageHttpError(400, `images supports at most ${MAX_INPUT_FILES} files`, "invalid_request");
    }
    masks = normalizeJsonImageList(b.masks !== undefined ? b.masks : b.mask, "mask");
  }

  return {
    model,
    prompt,
    n: parseN(b.n),
    size: String(b.size || "").trim(),
    quality: String(b.quality || "auto").trim() || "auto",
    background: String(b.background || "auto").trim() || "auto",
    output_format: outputFormat,
    frame_count: parseFrameCount(b.frame_count),
    frame_duration_ms: parseFrameDuration(b.frame_duration_ms),
    images,
    masks,
  };
}

export async function runWebImageWorker(payload, {
  python = workerPython,
  signal,
  timeout = 240_000,
  workerPath = IMAGE_WORKER,
} = {}) {
  if (signal?.aborted) throw abortError();
  const command = await python();
  const serialized = JSON.stringify(payload);

  return new Promise((resolve, reject) => {
    const child = spawn(command, [workerPath], {
      stdio: ["pipe", "pipe", "pipe"],
      shell: false,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
      },
    });

    const stdoutChunks = [];
    let stdoutBytes = 0;
    let stderr = "";
    let ended = false;
    let aborted = false;

    const cleanup = () => {
      clearTimeout(timer);
      if (signal && onAbort) {
        signal.removeEventListener("abort", onAbort);
      }
    };

    const finish = (err, output) => {
      if (ended) return;
      ended = true;
      cleanup();
      if (err) reject(err);
      else resolve(output);
    };

    const onAbort = () => {
      aborted = true;
      try { child.kill(); } catch {}
      finish(abortError());
    };

    if (signal) {
      signal.addEventListener("abort", onAbort, { once: true });
    }

    const timer = setTimeout(() => {
      try { child.kill(); } catch {}
      finish(null, {
        ok: false,
        http_status: 504,
        error_code: "worker_timeout",
        error: "Image generation worker timed out",
      });
    }, timeout);
    timer.unref?.();

    child.once("error", () => {
      finish(null, {
        ok: false,
        http_status: 502,
        error_code: "worker_start_failed",
        error: "Python image worker could not start",
      });
    });

    child.stdout.on("data", (chunk) => {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      stdoutBytes += buf.length;
      if (stdoutBytes > MAX_STDOUT_BYTES) {
        try { child.kill(); } catch {}
        finish(null, {
          ok: false,
          http_status: 502,
          error_code: "output_too_large",
          error: "Image worker output exceeded maximum size",
        });
        return;
      }
      stdoutChunks.push(buf);
    });

    child.stderr.on("data", (chunk) => {
      if (stderr.length < MAX_STDERR_BYTES) {
        stderr += String(chunk).slice(0, MAX_STDERR_BYTES - stderr.length);
      }
    });

    child.once("close", (code) => {
      if (ended) return;
      if (aborted) {
        finish(abortError());
        return;
      }
      const text = Buffer.concat(stdoutChunks).toString("utf8").trim();
      if (text) {
        try {
          finish(null, JSON.parse(text));
          return;
        } catch {}
      }
      finish(null, {
        ok: false,
        http_status: 502,
        error_code: code === 0 ? "invalid_worker_output" : "worker_crashed",
        error: "Image generation worker did not return valid JSON",
      });
    });

    child.stdin.on("error", () => {});
    child.stdin.end(serialized);
  });
}

function estimateUsage(request, dataItems) {
  const promptTokens = Math.max(1, Math.ceil(String(request.prompt || "").length / 4));
  const imageInputTokens = (request.images?.length || 0) * 512;
  const outputTokens = Math.max(1, dataItems.length) * 1024;
  return {
    input_tokens: promptTokens + imageInputTokens,
    output_tokens: outputTokens,
  };
}

export async function checkWebToken({
  accessToken,
  proxy = "",
  timezoneOffsetMin = -480,
  signal,
  runWorker = runWebImageWorker,
} = {}) {
  if (!accessToken || typeof accessToken !== "string") {
    return {
      ok: false,
      http_status: 401,
      error_code: "unauthorized",
      error: "Missing access_token",
    };
  }
  return runWorker(
    {
      action: "check_token",
      access_token: accessToken,
      proxy: proxy || "",
      timezone_offset_min: timezoneOffsetMin,
    },
    { signal, timeout: 30_000 },
  );
}

export async function checkWebImageQuota({
  accessToken,
  proxy = "",
  timezoneOffsetMin = -480,
  signal,
  runWorker = runWebImageWorker,
} = {}) {
  if (!accessToken || typeof accessToken !== "string") {
    return {
      ok: false,
      http_status: 401,
      error_code: "unauthorized",
      error: "Missing access_token",
    };
  }
  return runWorker(
    {
      action: "check_image_quota",
      access_token: accessToken,
      proxy: proxy || "",
      timezone_offset_min: timezoneOffsetMin,
    },
    { signal, timeout: 45_000 },
  );
}

export async function executeWebImageGeneration({
  accounts,
  request,
  signal,
  reauthenticate,
  runWorker = runWebImageWorker,
  proxy = "",
  workerOptions = {},
  maxTotalMs = 240_000,
  maxSlowFailovers = 3,
  onAccountResult = null,
} = {}) {
  if (!Array.isArray(accounts) || !accounts.length) {
    throw new ImageHttpError(503, "GPTFree pool is empty; import accounts first", "pool_empty");
  }
  if (signal?.aborted) throw abortError();

  async function reportResult(payload) {
    if (typeof onAccountResult !== "function") return;
    try {
      await onAccountResult(payload);
    } catch {}
  }

  const startedAt = Date.now();
  const poolAccounts = accounts.map((a) => ({ ...a }));
  const inflightReauth = new Map();

  async function reauthSlot(acctIdx, attemptedToken) {
    const current = poolAccounts[acctIdx];
    if (current?.accessToken && attemptedToken && current.accessToken !== attemptedToken &&
        accessTokenState(current.accessToken).state === "unexpired_unverified") {
      return current;
    }
    const key = current?.email || String(acctIdx);
    if (!inflightReauth.has(key)) {
      const p = Promise.resolve()
        .then(() => reauthenticate(current))
        .then((updated) => {
          if (updated?.accessToken) {
            poolAccounts[acctIdx] = updated;
          }
          return updated;
        })
        .finally(() => {
          inflightReauth.delete(key);
        });
      inflightReauth.set(key, p);
    }
    return inflightReauth.get(key);
  }

  async function generateSingle(startOffset, taskSignal) {
    let lastError = null;
    let slowFailures = 0;
    for (let offset = 0; offset < poolAccounts.length; offset++) {
      if (taskSignal?.aborted) throw abortError();
      if (offset > 0 && (Date.now() - startedAt >= maxTotalMs || slowFailures >= maxSlowFailovers)) {
        break;
      }
      const acctIdx = (startOffset + offset) % poolAccounts.length;
      let account = poolAccounts[acctIdx];
      if (accessTokenState(account.accessToken).state !== "unexpired_unverified") {
        try {
          account = await reauthSlot(acctIdx, account.accessToken);
        } catch (err) {
          if (err?.name === "AbortError" || taskSignal?.aborted) throw abortError();
          lastError = new ImageHttpError(401, String(err?.message || "Account reauthentication failed").slice(0, 180), "unauthorized");
          await reportResult({
            account,
            success: false,
            httpStatus: 401,
            errorCode: "unauthorized",
            rateLimited: false,
            retryAfter: null,
          });
          continue;
        }
      }

      for (let attempt = 0; attempt < 2; attempt++) {
        if (taskSignal?.aborted) throw abortError();
        const attemptStart = Date.now();
        const tokenUsed = account.accessToken;
        let res;
        try {
          res = await runWorker(
            {
              action: "generate_image",
              access_token: tokenUsed,
              prompt: request.prompt,
              model: request.model,
              model_id: request.model,
              size: request.size || "",
              quality: request.quality || "auto",
              output_format: request.output_format || "",
              background: request.background || "auto",
              frame_count: request.frame_count,
              frame_duration_ms: request.frame_duration_ms,
              images: request.images || [],
              masks: request.masks || [],
              proxy: proxy || "",
              ...workerOptions,
            },
            { signal: taskSignal },
          );
        } catch (err) {
          if (err?.name === "AbortError" || taskSignal?.aborted) throw abortError();
          lastError = new ImageHttpError(502, String(err?.message || "Image worker invocation failed").slice(0, 180), "worker_error");
          await reportResult({
            account,
            success: false,
            httpStatus: 502,
            errorCode: "worker_error",
            rateLimited: false,
            retryAfter: null,
          });
          if (Date.now() - attemptStart >= 10_000) slowFailures++;
          break;
        }

        if (res?.ok === true && Array.isArray(res.data) && res.data.length) {
          await reportResult({
            account,
            success: true,
            count: 1,
            httpStatus: 200,
          });
          return res;
        }

        const status = Number(res?.http_status || 502);
        const message = String(res?.error || "Upstream image generation failed").slice(0, 240);
        const code = String(res?.error_code || "upstream_error");

        if (status === 400) {
          await reportResult({
            account,
            success: false,
            httpStatus: 400,
            errorCode: code || "invalid_request",
            rateLimited: false,
            retryAfter: null,
          });
          throw new ImageHttpError(400, message, code || "invalid_request");
        }

        if (status === 401 && attempt === 0) {
          try {
            account = await reauthSlot(acctIdx, tokenUsed);
            continue;
          } catch (err) {
            if (err?.name === "AbortError" || taskSignal?.aborted) throw abortError();
            lastError = new ImageHttpError(401, String(err?.message || message).slice(0, 180), "unauthorized");
            await reportResult({
              account,
              success: false,
              httpStatus: 401,
              errorCode: "unauthorized",
              rateLimited: false,
              retryAfter: null,
            });
            break;
          }
        }

        lastError = new ImageHttpError(status, message, code);
        await reportResult({
          account,
          success: false,
          httpStatus: status,
          errorCode: code,
          rateLimited: status === 429 || code === "rate_limited",
          retryAfter: res?.retry_after ?? null,
        });
        if (![401, 403, 429, 500, 502, 503, 504].includes(status)) {
          throw lastError;
        }
        if ([500, 502, 504].includes(status) && (Date.now() - attemptStart >= 10_000)) {
          slowFailures++;
        }
        break;
      }
    }

    if (lastError && lastError.status === 400) throw lastError;
    throw new ImageHttpError(
      503,
      lastError?.message ? `所有 GPTFree 账号生图失败（最近原因：${lastError.message}）` :
        "所有 GPTFree 账号均不可用于当前生图请求；请检查账号状态、额度或网络代理。",
      lastError?.code || "pool_exhausted",
    );
  }

  const count = Math.max(1, Math.min(4, Number(request.n || 1)));
  if (count === 1) {
    const single = await generateSingle(0, signal);
    const data = single.data.slice(0, 1);
    return {
      created: Number(single.created) || Math.floor(Date.now() / 1000),
      data,
      usage: estimateUsage(request, data),
    };
  }

  const groupAbort = new AbortController();
  const forwardAbort = () => groupAbort.abort();
  signal?.addEventListener("abort", forwardAbort, { once: true });

  try {
    const settled = await Promise.all(
      Array.from({ length: count }, (_, idx) => generateSingle(idx % accounts.length, groupAbort.signal)),
    );
    const data = [];
    for (const item of settled) {
      if (Array.isArray(item?.data) && item.data.length) {
        data.push(item.data[0]);
      }
    }
    const finalData = data.slice(0, count);
    if (!finalData.length) {
      throw new ImageHttpError(502, "No images were returned by upstream", "no_image_generated");
    }
    return {
      created: Number(settled[0]?.created) || Math.floor(Date.now() / 1000),
      data: finalData,
      usage: estimateUsage(request, finalData),
    };
  } catch (err) {
    groupAbort.abort();
    throw err;
  } finally {
    signal?.removeEventListener("abort", forwardAbort);
  }
}
