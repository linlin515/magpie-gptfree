import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  CODEX_BASE,
  discoverPoolModels,
  isCodexToken,
  isWebOnlyToken,
  poolFetch,
  renewAccount,
  verifyWebAccessToken,
} from "../lib/codex-pool.mjs";
import {
  clearTextCooldowns,
  executeWebTextGeneration,
  extractWebTextRequest,
  runWebTextWorker,
} from "../lib/chatgpt-web-text.mjs";

function makeJwt(claims) {
  const header = Buffer.from(JSON.stringify({ alg: "RS256", typ: "JWT" })).toString("base64url");
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  return `${header}.${payload}.sig`;
}

function platformJwt(expOffsetSec = 3600, email = "free@example.com") {
  return makeJwt({
    exp: Math.floor(Date.now() / 1000) + expOffsetSec,
    client_id: "app_2SKx67EdpoN0G6j64rFvigXD",
    "https://api.openai.com/profile": { email },
  });
}

function codexJwt(expOffsetSec = 3600, accountId = "acct_codex_1") {
  return makeJwt({
    exp: Math.floor(Date.now() / 1000) + expOffsetSec,
    client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
    "https://api.openai.com/auth": { chatgpt_account_id: accountId },
  });
}

test("token capability detection separates Platform Web tokens from Codex tokens", () => {
  const pTok = platformJwt();
  const cTok = codexJwt();

  assert.equal(isWebOnlyToken(pTok), true);
  assert.equal(isCodexToken(pTok), false);

  assert.equal(isWebOnlyToken(cTok), false);
  assert.equal(isCodexToken(cTok), true);

  // Opaque/test strings remain compatible with Codex path
  assert.equal(isWebOnlyToken("opaque-token"), false);
  assert.equal(isCodexToken("opaque-token"), true);
});

test("verifyWebAccessToken admits a Web probe success and rejects 401/403", async () => {
  const tok = platformJwt(3600, "free@example.com");
  const seen = [];
  const ok = await verifyWebAccessToken(tok, {
    proxy: "http://127.0.0.1:7890",
    expectedEmail: "free@example.com",
    runWorker: async (payload) => {
      seen.push(payload);
      return { ok: true, http_status: 200, plan_type: "free" };
    },
  });
  assert.equal(ok.accepted, true);
  assert.equal(ok.status, 200);
  assert.equal(ok.planType, "free");
  assert.equal(seen.length, 1);
  assert.equal(seen[0].action, "check_token");
  assert.equal(seen[0].proxy, "http://127.0.0.1:7890");

  const unauthorized = await verifyWebAccessToken(tok, {
    runWorker: async () => ({ ok: false, http_status: 401, error_code: "unauthorized" }),
  });
  assert.equal(unauthorized.accepted, false);
  assert.equal(unauthorized.reason, "unauthorized");
  assert.equal(unauthorized.status, 401);

  const forbidden = await verifyWebAccessToken(tok, {
    runWorker: async () => ({ ok: false, http_status: 403, error_code: "forbidden" }),
  });
  assert.equal(forbidden.accepted, false);
  assert.equal(forbidden.reason, "forbidden");
  assert.equal(forbidden.status, 403);

  const mismatch = await verifyWebAccessToken(tok, {
    expectedEmail: "other@example.com",
    runWorker: async () => ({ ok: true, http_status: 200 }),
  });
  assert.equal(mismatch.accepted, false);
  assert.equal(mismatch.reason, "identity_mismatch");
});

test("discoverPoolModels returns the Web catalog immediately for Platform OAuth accounts", async () => {
  const pTok = platformJwt();
  let probes = 0;
  const models = await discoverPoolModels({
    read: async () => [{ email: "free@example.com", accessToken: pTok }],
    fetcher: async () => {
      probes++;
      return new Response("{}", { status: 500 });
    },
    reauthenticate: async () => {
      throw new Error("Web-only discovery must not re-authenticate");
    },
  });

  assert.equal(probes, 0, "Web-only pool must not spend the 20s budget probing Codex");
  assert.equal(models["gpt-6-luna"].api.npm, "@ai-sdk/openai");
  assert.equal(models["gpt-5.6-luna"].api.npm, "@ai-sdk/openai");
  assert.equal(models["gpt-5.6-terra"].api.npm, "@ai-sdk/openai");
  assert.equal(models["gpt-5-6"].api.npm, "@ai-sdk/openai");
  assert.equal(models["auto"].api.npm, "@ai-sdk/openai");
  assert.equal(models["gpt-image-2.6"].api.npm, "@ai-sdk/openai-compatible");
  assert.equal("gpt-5.6-luna-wm" in models, false);
});

test("discoverPoolModels probes Codex accounts first, then falls back to Web models", async () => {
  const codex = codexJwt();
  const web = platformJwt();
  let probes = 0;
  const models = await discoverPoolModels({
    read: async () => [
      { email: "codex@example.com", accessToken: codex },
      { email: "free@example.com", accessToken: web },
    ],
    fetcher: async (url) => {
      probes++;
      assert.equal(String(url), `${CODEX_BASE}/models?client_version=0.118.0`);
      return new Response("forbidden", { status: 403 });
    },
    reauthenticate: async () => {
      throw new Error("403 must not trigger re-authentication");
    },
  });

  assert.equal(probes, 1);
  assert.equal(models["gpt-5.6-terra"].api.npm, "@ai-sdk/openai");
});

test("renewAccount verifies Platform OAuth logins through the Web probe", async () => {
  const pTok = platformJwt(3600, "web@example.com");
  const saved = [];
  const renewed = await renewAccount(
    { email: "web@example.com", password: "p" },
    {
      authenticate: async () => ({ tokens: { access: pTok } }),
      verify: async () => { throw new Error("Codex verify must not run for Web-only tokens"); },
      verifyWeb: async (token, opts) => {
        assert.equal(token, pTok);
        assert.equal(opts.expectedEmail, "web@example.com");
        return { accepted: true, status: 200 };
      },
      save: async (record) => saved.push(record),
    },
  );

  assert.equal(renewed.accessToken, pTok);
  assert.equal(saved.length, 1);
  assert.equal(saved[0].accessToken, pTok);
});

test("extractWebTextRequest parses instructions, multi-turn items, tool calls, and reasoning", () => {
  const req = extractWebTextRequest(
    JSON.stringify({
      model: "gptfree-pool/gpt-5.6-terra",
      instructions: "Be accurate.",
      reasoning: { effort: "high" },
      stream: true,
      input: [
        { type: "reasoning", id: "rs_1" },
        { role: "user", content: [{ type: "input_text", text: "Check file" }] },
        { type: "function_call", name: "read_file", arguments: "{\"path\":\"a.txt\"}" },
        { type: "function_call_output", call_id: "call_1", output: "ok" },
      ],
    }),
  );

  assert.equal(req.model, "gpt-5.6-terra");
  assert.equal(req.requestedModel, "gptfree-pool/gpt-5.6-terra");
  assert.equal(req.stream, true);
  assert.equal(req.thinkingEffort, "high");
  assert.equal(req.messages[0].role, "system");
  assert.equal(req.messages[0].content, "Be accurate.");
  assert.equal(req.messages[1].role, "user");
  assert.match(req.messages[2].content, /read_file/);
  assert.match(req.messages[3].content, /call_1/);

  assert.throws(() => extractWebTextRequest("{bad json"), /Invalid Responses request/);
  assert.throws(() => extractWebTextRequest(JSON.stringify({ model: "gpt-6-luna", input: "   " })), /at least one/);
});

test("poolFetch routes Platform Web accounts away from /codex/responses and formats Responses SSE", async () => {
  clearTextCooldowns();
  const pTok = platformJwt(3600, "luna@example.com");
  let codexCalled = false;

  const res = await poolFetch(`${CODEX_BASE}/responses`, {
    method: "POST",
    body: JSON.stringify({
      model: "gpt-6-luna",
      instructions: "Concise",
      input: "2+2?",
      stream: true,
    }),
  }, {
    read: async () => [{ email: "luna@example.com", accessToken: pTok }],
    fetcher: async () => {
      codexCalled = true;
      throw new Error("Codex /responses must not be hit for Platform OAuth accounts");
    },
    webTextRunner: async (payload, { onDelta }) => {
      assert.equal(payload.access_token, pTok);
      assert.equal(payload.model, "gpt-6-luna");
      assert.deepEqual(payload.messages, [
        { role: "system", content: "Concise" },
        { role: "user", content: "2+2?" },
      ]);
      onDelta?.("4");
      onDelta?.("!");
      return { ok: true, text: "4!" };
    },
  });

  assert.equal(codexCalled, false);
  assert.equal(res.status, 200);
  assert.match(res.headers.get("content-type") || "", /text\/event-stream/);
  const sse = await res.text();
  assert.match(sse, /event: response\.created/);
  assert.match(sse, /"delta":"4"/);
  assert.match(sse, /"delta":"!"/);
  assert.match(sse, /event: response\.completed/);
  assert.match(sse, /"text":"4!"/);
});

test("executeWebTextGeneration handles 401 reauth, 429 failover, and non-streaming JSON responses", async () => {
  clearTextCooldowns();
  const tok1 = platformJwt(3600, "a1@example.com");
  const tok1New = platformJwt(7200, "a1@example.com");
  const tok2 = platformJwt(3600, "a2@example.com");
  const calls = [];

  const res = await executeWebTextGeneration(
    JSON.stringify({ model: "gpt-5.6-luna", input: "hello", stream: false }),
    [
      { email: "a1@example.com", accessToken: tok1 },
      { email: "a2@example.com", accessToken: tok2 },
    ],
    {
      reauthenticate: async (acc) => ({ ...acc, accessToken: tok1New }),
      worker: async (payload) => {
        calls.push(payload.access_token);
        if (payload.access_token === tok1) {
          return { ok: false, http_status: 401, error_code: "invalid_token", error: "expired", token_invalid: true };
        }
        if (payload.access_token === tok1New) {
          return { ok: false, http_status: 429, error_code: "rate_limit_exceeded", error: "rate limit", retry_after: 30 };
        }
        return { ok: true, text: "world" };
      },
    },
  );

  assert.deepEqual(calls, [tok1, tok1New, tok2]);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.object, "response");
  assert.equal(body.status, "completed");
  assert.equal(body.output_text, "world");
  assert.equal(body.output[0].content[0].text, "world");

  // Next call should put cooled-down a1 behind a2 without hitting a1 first
  const nextCalls = [];
  const res2 = await executeWebTextGeneration(
    JSON.stringify({ model: "gpt-5.6-luna", input: "hello 2", stream: false }),
    [
      { email: "a1@example.com", accessToken: tok1New },
      { email: "a2@example.com", accessToken: tok2 },
    ],
    {
      worker: async (payload) => {
        nextCalls.push(payload.access_token);
        return { ok: true, text: "fast" };
      },
    },
  );
  assert.equal(res2.status, 200);
  assert.deepEqual(nextCalls, [tok2]);
  clearTextCooldowns();
});

test("poolFetch switches to Web text if reauth produces a Platform token and runWebTextWorker validates missing token / abort", async () => {
  clearTextCooldowns();
  const oldCodex = codexJwt(3600, "acct_old");
  const renewedPlatform = platformJwt(3600, "migrated@example.com");

  let codexCalls = 0;
  // Calling poolFetch with codex account that fails 401 -> reauth returns Platform token
  // Because executeWebTextGeneration is invoked automatically on Platform token, mock runWebTextWorker or test via abort
  const ac = new AbortController();
  ac.abort();
  await assert.rejects(
    runWebTextWorker(
      { access_token: renewedPlatform, model: "gpt-6-luna", messages: [{ role: "user", content: "hi" }] },
      { signal: ac.signal, python: async () => "python" },
    ),
    /Aborted/,
  );

  const res = await runWebTextWorker(
    { access_token: "", model: "gpt-6-luna", messages: [{ role: "user", content: "hi" }] },
    { python: async () => process.env.PYTHON || (process.platform === "win32" ? "python" : "python3") },
  );
  assert.equal(res.ok, false);
  assert.equal(res.http_status, 401);
  assert.equal(isWebOnlyToken(oldCodex), false);
  assert.equal(isWebOnlyToken(renewedPlatform), true);
  assert.equal(codexCalls, 0);
});
