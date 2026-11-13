import { readFile } from "node:fs/promises";
import { parseAccounts, safeAccountName, validateProxyUrl } from "./credentials.mjs";
import { loginHTTPAccount } from "./dsh-http-auth.mjs";
import { probeAccountImageQuota, isWebOnlyToken, verifyAccessToken, verifyWebAccessToken } from "./codex-pool.mjs";
import { accountSlotTag, formatAccountSlotLabel } from "./pool-connection.mjs";
import { defaultStatsFile } from "./pool-stats.mjs";
import { upsertPool } from "./pool-vault.mjs";

/**
 * DSH-style HTTP authentication with a separate Magpie token pool.
 * No Codex CLI command, no browser launch, no DSH service dependency.
 */
export async function runBatch({
  file, consent = false, dryRun = false, proxy = "", skipInvalid = false,
  mode = "pool", onProgress = () => {}, signal,
  read = readFile, login = loginHTTPAccount,
  verifyToken = verifyAccessToken, verifyWebToken = verifyWebAccessToken,
  savePool = upsertPool,
  checkQuota, statsFile, statsStore,
} = {}) {
  if (!file || typeof file !== "string") throw new Error("请指定 TXT 文件的完整路径");
  if (mode !== "pool") throw new Error("已移除 Codex CLI 原生导入，请使用 GPT号池");
  const raw = await read(file, "utf8");
  const parsed = parseAccounts(raw);
  if (dryRun) {
    return {
      total: parsed.accounts.length, duplicates: parsed.duplicates,
      invalid: parsed.errors, added: 0, failed: [], mode,
      ready: parsed.accounts.map((a) => ({
        line: a.line, email: safeAccountName(a.email),
        has2FA: !!a.totp, kind: a.kind,
        tokenState: a.kind === "access_token" ? a.tokenState : undefined,
      })),
    };
  }
  if (!consent) throw new Error("导入真实账号前必须确认账号的使用授权");
  if (!parsed.accounts.length) throw new Error("TXT 中没有有效账号");
  if (parsed.errors.length && !skipInvalid) throw new Error("账号列表存在格式错误，请先检查导入预览");
  const proxyCheck = validateProxyUrl(proxy);
  if (proxyCheck.error) throw new Error(proxyCheck.error);
  const safeProxy = proxyCheck.proxy;
  const quotaChecker = checkQuota !== undefined
    ? checkQuota
    : (savePool === upsertPool && verifyToken === verifyAccessToken && login === loginHTTPAccount
      ? probeAccountImageQuota
      : null);
  const quotaTarget = (statsFile !== undefined || statsStore !== undefined)
    ? {
        ...(statsFile !== undefined ? { statsFile } : {}),
        ...(statsStore !== undefined ? { statsStore } : {}),
      }
    : (savePool === upsertPool ? { statsFile: defaultStatsFile() } : { statsFile: null });
  const result = {
    total: parsed.accounts.length, added: 0, reused: 0,
    mode, failed: [], duplicates: parsed.duplicates,
    invalid: skipInvalid ? parsed.errors.slice(0, 100).map(({ line, reason }) => ({ line, reason })) : [],
    formats: { standard: 0, mail_link: 0, access_token: 0 },
    tokenNeedsOAuth: 0,
  };
  for (const account of parsed.accounts) {
    if (signal?.aborted) throw new Error("批量导入已取消");
    result.formats[account.kind]++;
    const safe = { email: safeAccountName(account.email), line: account.line };
    const slotMeta = { slotTag: accountSlotTag(account.email), accountLabel: formatAccountSlotLabel(account.email) };
    if (account.kind === "mail_link") {
      // Additional mailbox URL is format metadata, not fetched by the importer.
      onProgress({ state: "mail_link_ignored", ...safe });
    }
    try {
      if (account.kind === "access_token") {
        if (account.tokenState === "unexpired_unverified") {
          const probe = isWebOnlyToken(account.accessToken)
            ? await verifyWebToken(account.accessToken, {
                expectedEmail: account.email, proxy: safeProxy, signal,
              })
            : await verifyToken(account.accessToken, {
                expectedEmail: account.email, proxy: safeProxy, signal,
              });
          if (probe.accepted) {
            const savedRecord = {
              email: account.email, password: account.password,
              totp: account.totp, accessToken: account.accessToken,
            };
            await savePool(savedRecord);
            result.added++;
            result.reused++;
            onProgress({ state: "token_reused", ...safe, ...slotMeta });
            if (typeof quotaChecker === "function") {
              await quotaChecker(savedRecord, {
                proxy: safeProxy, signal, ...quotaTarget, reauthenticate: null,
              }).catch(() => {});
            }
            continue;
          }
        }
        result.tokenNeedsOAuth++;
        onProgress({
          state: account.tokenState === "expired" ? "token_expired" : "token_needs_oauth",
          ...safe,
        });
      }
      onProgress({ state: "authorizing", ...safe });
      // DSH Platform HTTP OAuth login, not Codex CLI PKCE or Playwright.
      const { tokens } = await login(account, { signal, proxy: safeProxy });
      // Platform and Codex tokens are not automatically interchangeable.
      // Platform OAuth tokens are Web-only; Codex tokens must still pass the
      // live Codex /models probe. Both checks honour the configured proxy.
      const probe = isWebOnlyToken(tokens.access)
        ? await verifyWebToken(tokens.access, {
            expectedEmail: account.email, proxy: safeProxy, signal,
          })
        : await verifyToken(tokens.access, {
            expectedEmail: account.email, proxy: safeProxy, signal,
          });
      if (!probe.accepted) {
        const detail = probe.reason ? `（${String(probe.reason).slice(0, 80)}）` : "";
        throw new Error("DSH HTTP 登录成功，但 Platform Token 未通过 Magpie 模型访问验证" + detail);
      }
      const savedRecord = {
        email: account.email, password: account.password,
        totp: account.totp, accessToken: tokens.access,
      };
      await savePool(savedRecord);
      result.added++;
      onProgress({ state: "added", ...safe, ...slotMeta });
      if (typeof quotaChecker === "function") {
        await quotaChecker(savedRecord, {
          proxy: safeProxy, signal, ...quotaTarget, reauthenticate: null,
        }).catch(() => {});
      }
    } catch (err) {
      if (signal?.aborted) throw new Error("批量导入已取消");
      // Do not relay arbitrary upstream exceptions that may contain auth data.
      const message = String(err?.message || "");
      const known = /DSH HTTP|Platform Token|没有返回可用|未找到可用 Python|HTTP 登录|认证后端/.test(message);
      const reason = known ? message.slice(0, 180) :
        "后台 HTTP 授权失败；请检查账号、TOTP、网络或代理。不会自动打开登录网页";
      result.failed.push({ line: account.line, email: safeAccountName(account.email), reason });
      onProgress({ state: "failed", ...safe, reason });
    }
  }
  return result;
}
