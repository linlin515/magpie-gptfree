import { strict as assert } from "node:assert";
import { test } from "node:test";
import { runBatch } from "../lib/importer.mjs";
import { normalizeWorkerResult, loginHTTPAccount } from "../lib/dsh-http-auth.mjs";
import { GPTFreePoolPlugin } from "../index.mjs";

const seed = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ";
const jwt = (exp) => [
  Buffer.from('{"alg":"RS256","typ":"JWT"}').toString("base64url"),
  Buffer.from(JSON.stringify({ exp })).toString("base64url"), "signature",
].join(".");

const platformJwt = (email = "free@example.com", expSeconds = 3600) => [
  Buffer.from('{"alg":"RS256","typ":"JWT"}').toString("base64url"),
  Buffer.from(JSON.stringify({
    exp: Math.floor(Date.now() / 1000) + expSeconds,
    client_id: "app_2SKx67EdpoN0G6j64rFvigXD",
    "https://api.openai.com/profile": { email },
  })).toString("base64url"),
  "signature",
].join(".");

test("dry-run parses three import formats, without contacting upstream", async () => {
  const content = [
    `a@example.com----p1----${seed}`,
    `b@example.com---p2---${seed}---https://mail.xxx.store/abcd`,
    `c@example.com----p3----${seed}----${jwt(Math.floor(Date.now()/1000)+3600)}`,
  ].join("\n");
  const result = await runBatch({
    file: "virtual.txt", dryRun: true, read: async () => content,
    login: () => { throw Error("No authentication during dry-run"); },
  });
  assert.equal(result.total, 3);
  assert.deepEqual(result.ready.map((x) => x.kind), ["standard", "mail_link", "access_token"]);
  assert.equal(result.ready[2].tokenState, "unexpired_unverified");
  assert(!JSON.stringify(result).includes("p3"));
  assert(!JSON.stringify(result).includes("mail.xxx.store"));
});

test("all TXT records use the standalone DSH HTTP authentication path", async () => {
  const future = jwt(Math.floor(Date.now()/1000)+3600);
  const expired = jwt(Math.floor(Date.now()/1000)-3600);
  const secretLink = "https://mail.xxx.store/key---secret";
  const content = [
    `a@example.com----password1----${seed}`,
    `b@example.com----password2----${seed}----${secretLink}`,
    `c@example.com----password3----${seed}----${future}`,
    `d@example.com---password4---${seed}---${expired}`,
  ].join("\n");
  const calls = [], states = [], stored = [];
  const result = await runBatch({
    file: "virtual.txt", read: async () => content, consent: true,
    login: async (account) => {
      calls.push(account.email);
      return { tokens: { access: "fresh", refresh: "r", id: "i" } };
    },
    verifyToken: async (token) => ({ accepted: token === "fresh" }),
    savePool: async (account) => { stored.push(account); },
    onProgress: (event) => states.push(event),
  });
  assert.equal(result.added, 4);
  assert.equal(result.reused, 0);
  assert.equal(result.tokenNeedsOAuth, 2);
  assert.equal(stored.length, 4);
  assert.deepEqual(result.formats, { standard: 1, mail_link: 1, access_token: 2 });
  assert.deepEqual(calls, ["a@example.com", "b@example.com", "c@example.com", "d@example.com"]);
  assert(states.some((x) => x.state === "mail_link_ignored"));
  assert(states.some((x) => x.state === "token_needs_oauth"));
  assert(states.some((x) => x.state === "token_expired"));
  const printed = JSON.stringify(states);
  for (const x of ["password1", secretLink, future, expired, seed])
    assert(!printed.includes(x));
});

test("skipInvalid imports valid multi-account batch with proxy and reports invalid lines safely", async () => {
  const txt = [
    `bad-email----pw----${seed}`,
    `good1@example.com----pw1----${seed}`,
    `good2@example.com----pw2----${seed}`,
  ].join("\n");
  await assert.rejects(runBatch({
    file: "virtual.txt", read: async () => txt, consent: true,
  }), /格式错误/);
  const seenProxies = [];
  const stored = [];
  const result = await runBatch({
    file: "virtual.txt",
    read: async () => txt,
    consent: true,
    skipInvalid: true,
    proxy: "http://127.0.0.1:7890",
    login: async (account, opts) => {
      seenProxies.push(opts.proxy);
      return { tokens: { access: "fresh-" + account.email, refresh: "", id: "" } };
    },
    verifyToken: async () => ({ accepted: true }),
    savePool: async (record) => stored.push(record.email),
  });
  assert.equal(result.total, 2);
  assert.equal(result.added, 2);
  assert.equal(result.invalid.length, 1);
  assert.equal(result.invalid[0].line, 1);
  assert.deepEqual(seenProxies, ["http://127.0.0.1:7890", "http://127.0.0.1:7890"]);
  assert.deepEqual(stored, ["good1@example.com", "good2@example.com"]);
});

test("one failed HTTP account does not discard the next one or leak exceptions", async () => {
  const txt = [`a@example.com----p----${seed}`, `b@example.com----p----${seed}`].join("\n");
  const result = await runBatch({
    file: "virtual.txt", read: async () => txt, consent: true,
    login: async (account) => {
      if (account.email === "a@example.com") throw Error("token=secret-in-exception");
      return { tokens: { access: "fresh", refresh: "r", id: "i" } };
    },
    verifyToken: async () => ({ accepted: true }),
    savePool: async () => {},
  });
  assert.equal(result.added, 1);
  assert.equal(result.failed.length, 1);
  assert(!JSON.stringify(result).includes("secret-in-exception"));
});

test("DSH Platform token is not falsely marked usable without model verification", async () => {
  const txt = `a@example.com----pw----${seed}`;
  const stored = [];
  const result = await runBatch({
    file: "virtual.txt", read: async () => txt, consent: true,
    login: async () => ({ tokens: { access: "platform-token", id: "", refresh: "r" } }),
    verifyToken: async () => ({ accepted: false, reason: "status_401" }),
    savePool: async (item) => stored.push(item),
  });
  assert.equal(result.added, 0);
  assert.equal(result.failed.length, 1);
  assert.equal(stored.length, 0);
  assert.match(result.failed[0].reason, /Platform Token/);
});

test("Platform web token is admitted through the web probe with proxy forwarded", async () => {
  const txt = `a@example.com----pw----${seed}`;
  const token = platformJwt("a@example.com");
  const seen = [], stored = [];
  const result = await runBatch({
    file: "virtual.txt", read: async () => txt, consent: true,
    proxy: "http://127.0.0.1:7890",
    login: async () => ({ tokens: { access: token, id: "", refresh: "r" } }),
    verifyToken: async () => { throw new Error("Codex probe must not run for web-only tokens"); },
    verifyWebToken: async (value, opts) => {
      seen.push({ value, opts });
      return { accepted: true, status: 200 };
    },
    savePool: async (item) => stored.push(item),
  });
  assert.equal(result.added, 1);
  assert.equal(stored.length, 1);
  assert.equal(stored[0].accessToken, token);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].value, token);
  assert.equal(seen[0].opts.proxy, "http://127.0.0.1:7890");
  assert.equal(seen[0].opts.expectedEmail, "a@example.com");
});

test("Platform web token rejected by the web probe is not admitted", async () => {
  const txt = `a@example.com----pw----${seed}`;
  const stored = [];
  const result = await runBatch({
    file: "virtual.txt", read: async () => txt, consent: true,
    proxy: "http://127.0.0.1:7890",
    login: async () => ({ tokens: { access: platformJwt("a@example.com"), id: "", refresh: "r" } }),
    verifyToken: async () => ({ accepted: true }),
    verifyWebToken: async () => ({ accepted: false, reason: "status_403", status: 403 }),
    savePool: async (item) => stored.push(item),
  });
  assert.equal(result.added, 0);
  assert.equal(stored.length, 0);
  assert.equal(result.failed.length, 1);
  assert.match(result.failed[0].reason, /status_403/);
});

test("reusable Platform web access token is checked by the web probe", async () => {
  const token = platformJwt("reuse@example.com");
  const txt = `reuse@example.com----pw----${seed}----${token}`;
  let loginCalls = 0, webProbes = 0;
  const stored = [];
  const result = await runBatch({
    file: "virtual.txt", read: async () => txt, consent: true,
    login: async () => { loginCalls++; return { tokens: { access: "fresh", id: "", refresh: "r" } }; },
    verifyToken: async () => ({ accepted: false }),
    verifyWebToken: async () => { webProbes++; return { accepted: true, status: 200 }; },
    savePool: async (item) => stored.push(item),
  });
  assert.equal(result.added, 1);
  assert.equal(result.reused, 1);
  assert.equal(loginCalls, 0);
  assert.equal(webProbes, 1);
  assert.equal(stored[0].accessToken, token);
});

test("Codex CLI mode is disabled", async () => {
  await assert.rejects(runBatch({
    file: "virtual.txt", mode: "native", read: async () => "a@example.com----pw",
  }), /已移除 Codex CLI/);
});

test("DSH worker response is normalized and failures reject", async () => {
  assert.deepEqual(normalizeWorkerResult('{"ok":true,"access_token":"platform","refresh_token":"r"}'),
    { access: "platform", refresh: "r", id: "" });
  assert.throws(() => normalizeWorkerResult('{"ok":false,"error":"bad_password"}'), /bad_password/);
  assert.throws(() => normalizeWorkerResult("not json"), /无效数据/);
});

test("bridge sends login credentials through stdin, not process argv", async () => {
  let command, args, options;
  const result = await loginHTTPAccount({
    email: "a@example.com", password: "never-in-argv", totp: seed,
  }, {
    python: async () => "/private/python",
    worker: async (binary, parameters, opts) => {
      command = binary; args = parameters; options = opts;
      return '{"ok":true,"access_token":"platform-token"}';
    },
  });
  assert.equal(command, "/private/python");
  assert.equal(args.length, 1);
  assert(!JSON.stringify(args).includes("never-in-argv"));
  assert.equal(JSON.parse(options.input).password, "never-in-argv");
  assert.equal(result.tokens.access, "platform-token");
});

test("Magpie provider still exposes its native in-app import entry", async () => {
  const plug = await GPTFreePoolPlugin();
  const config = { provider: {} };
  await plug.config(config);
  assert("gptfree-pool" in config.provider);
  assert.equal(plug.auth.methods.length, 1);
  assert.equal(plug.auth.methods[0].type, "oauth");
});

test("runBatch invokes checkQuota best-effort after saving & reporting progress without failing batch", async () => {
  const future = jwt(Math.floor(Date.now() / 1000) + 3600);
  const content = [
    `q1@example.com----pw1----${seed}`,
    `q2@example.com----pw2----${seed}----${future}`,
  ].join("\n");
  const events = [];
  const quotaCalls = [];
  const res = await runBatch({
    file: "virtual.txt",
    read: async () => content,
    consent: true,
    login: async () => ({ tokens: { access: future, refresh: "", id: "" } }),
    verifyToken: async () => ({ accepted: true }),
    savePool: async () => {},
    onProgress: (ev) => {
      if (ev.state === "added" || ev.state === "token_reused") {
        events.push(`progress:${ev.line}`);
      }
    },
    checkQuota: async (record) => {
      quotaCalls.push(record.email);
      events.push(`quota:${record.email}`);
      if (record.email === "q1@example.com") {
        throw new Error("transient chatgpt.com timeout");
      }
    },
  });
  assert.equal(res.added, 2);
  assert.equal(res.failed.length, 0);
  assert.deepEqual(quotaCalls, ["q1@example.com", "q2@example.com"]);
  assert.deepEqual(events, [
    "progress:1",
    "quota:q1@example.com",
    "progress:2",
    "quota:q2@example.com",
  ]);
});
