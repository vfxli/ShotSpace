import React, { useCallback, useEffect, useState } from "react";
import { api, setAdminToken } from "../api";
import { AdminMachine, AdminOverview, AdminTaskOwner, QueueJobInfo } from "../types";

interface Props {
  onNavigateWorkbench: () => void;
}

/** "2026-11-24T08:12:03+0800" -> "11-24 08:12" */
function shortTime(iso: string): string {
  const m = /^\d{4}-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso || "");
  return m ? `${m[1]}-${m[2]} ${m[3]}:${m[4]}` : iso;
}

export const AdminPage: React.FC<Props> = ({ onNavigateWorkbench }) => {
  const [loading, setLoading] = useState(true);
  const [initialized, setInitialized] = useState(true);
  const [authed, setAuthed] = useState(false);

  const [setupPort, setSetupPort] = useState<number>(8765);
  const [setupCacheDir, setSetupCacheDir] = useState("");
  const [defaultCacheDir, setDefaultCacheDir] = useState("");
  const [setupPwd, setSetupPwd] = useState("");
  const [setupPwd2, setSetupPwd2] = useState("");
  const [loginPwd, setLoginPwd] = useState("");
  const [authErr, setAuthErr] = useState<string | null>(null);
  const [appVersion, setAppVersion] = useState<string>("");

  useEffect(() => {
    api
      .health()
      .then((h) => setAppVersion(h.version || ""))
      .catch(() => undefined);
  }, []);

  const [overview, setOverview] = useState<AdminOverview | null>(null);
  // v0.2.12: every task with the computer it belongs to, and every computer with how many tasks it has
  const [taskOwners, setTaskOwners] = useState<AdminTaskOwner[]>([]);
  const [machines, setMachines] = useState<AdminMachine[]>([]);
  const [ownerSearch, setOwnerSearch] = useState("");
  const [ownerFilter, setOwnerFilter] = useState<string | null>(null);
  const [ownerBusy, setOwnerBusy] = useState<string | null>(null);
  const [confirmDeleteTask, setConfirmDeleteTask] = useState<string | null>(null);
  const [assignOtherFor, setAssignOtherFor] = useState<string | null>(null);
  const [assignOtherIp, setAssignOtherIp] = useState("");

  const fetchOwners = useCallback(async () => {
    try {
      const [own, mach] = await Promise.all([api.adminTaskOwners(), api.adminMachines()]);
      setTaskOwners(own.shots);
      setMachines(mach.machines);
    } catch {
      // the overview reports a lost login
    }
  }, []);

  const runOwnerAction = async (taskId: string, action: () => Promise<unknown>) => {
    setOwnerBusy(taskId);
    try {
      await action();
      setAssignOtherFor(null);
      setConfirmDeleteTask(null);
      await fetchOwners();
    } catch (err: any) {
      setNotice({ type: "err", text: err?.message || String(err) });
    } finally {
      setOwnerBusy(null);
    }
  };

  const shownOwners = taskOwners.filter((t) => {
    if (ownerFilter === "__unclassified" && !t.is_public) return false;
    if (ownerFilter && ownerFilter !== "__unclassified" && t.owner_key !== ownerFilter) return false;
    const q = ownerSearch.trim().toLowerCase();
    return !q || t.name.toLowerCase().includes(q) || t.owner_label.toLowerCase().includes(q);
  });
  const [notice, setNotice] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  const [oldPwd, setOldPwd] = useState("");
  const [newPwd, setNewPwd] = useState("");
  const [newPwd2, setNewPwd2] = useState("");
  const [cfgPort, setCfgPort] = useState<number>(8765);
  const [cacheDirInput, setCacheDirInput] = useState<string>("");
  const [cacheDirTouched, setCacheDirTouched] = useState(false);
  const [savingCacheDir, setSavingCacheDir] = useState(false);
  const [retentionDays, setRetentionDays] = useState<number>(30);
  const [retentionTouched, setRetentionTouched] = useState(false);
  const [portTouched, setPortTouched] = useState(false);

  const fetchOverview = useCallback(async () => {
    try {
      const data = await api.adminOverview();
      setOverview(data);
      if (data.server?.port && !portTouched) setCfgPort(data.server.port);
      if (data.server?.cache_dir && !cacheDirTouched) {
        setCacheDirInput(data.server.cache_dir);
      }
      if (data.server?.retention_days !== undefined && !retentionTouched) {
        setRetentionDays(data.server.retention_days);
      }
    } catch (e: any) {
      if (e.status === 401) {
        setAdminToken("");
        setAuthed(false);
      }
    }
  }, [portTouched, cacheDirTouched, retentionTouched]);

  useEffect(() => {
    (async () => {
      try {
        const st = await api.adminStatus();
        setInitialized(st.initialized);
        setSetupPort(st.port || 8765);
        setCfgPort(st.port || 8765);
        if (st.default_cache_dir) setDefaultCacheDir(st.default_cache_dir);
        if (st.initialized && st.authenticated) {
          setAuthed(true);
          await fetchOverview();
        }
      } catch (e: any) {
        setAuthErr(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [fetchOverview]);

  useEffect(() => {
    if (!authed) return;
    const timer = window.setInterval(fetchOverview, 2000);
    return () => window.clearInterval(timer);
  }, [authed, fetchOverview]);

  useEffect(() => {
    if (!authed) return;
    fetchOwners();
    const timer = window.setInterval(fetchOwners, 5000);
    return () => window.clearInterval(timer);
  }, [authed, fetchOwners]);

  const handleFirstSetup = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthErr(null);
    if (setupPwd.length < 4) {
      setAuthErr("后台密码长度不能少于 4 位");
      return;
    }
    if (setupPwd !== setupPwd2) {
      setAuthErr("两次输入的密码不一致");
      return;
    }
    try {
      const res = await api.adminSetup(setupPort, setupPwd, setupCacheDir.trim());
      setAdminToken(res.token);
      setInitialized(true);
      setAuthed(true);
      await fetchOverview();
    } catch (err: any) {
      setAuthErr(err.message);
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthErr(null);
    try {
      const res = await api.adminLogin(loginPwd);
      setAdminToken(res.token);
      setLoginPwd("");
      setAuthed(true);
      await fetchOverview();
    } catch (err: any) {
      setAuthErr(err.message);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setNotice(null);
    if (newPwd.length < 4) {
      setNotice({ type: "err", text: "新密码长度不能少于 4 位" });
      return;
    }
    if (newPwd !== newPwd2) {
      setNotice({ type: "err", text: "两次输入的新密码不一致" });
      return;
    }
    try {
      const res = await api.adminChangePassword(oldPwd, newPwd);
      setAdminToken(res.token);
      setOldPwd("");
      setNewPwd("");
      setNewPwd2("");
      setNotice({ type: "ok", text: "后台管理密码已更新" });
    } catch (err: any) {
      setNotice({ type: "err", text: err.message });
    }
  };

  const handleSavePort = async (e: React.FormEvent) => {
    e.preventDefault();
    setNotice(null);
    try {
      const res = await api.adminSetPort(cfgPort);
      setPortTouched(false);
      setNotice({
        type: "ok",
        text: `默认启动端口已保存为 ${res.port}（下次重启服务时生效）`,
      });
    } catch (err: any) {
      setNotice({ type: "err", text: err.message });
    }
  };

  const handleSaveRetention = async (days: number) => {
    setNotice(null);
    try {
      const res = await api.adminSetRetention(days, true);
      setRetentionDays(res.retention_days);
      setRetentionTouched(false);
      await fetchOverview();
      setNotice({
        type: "ok",
        text:
          res.retention_days === 0
            ? "解算数据保留策略已设为：永久保留"
            : `解算数据保留期限已设为 ${res.retention_days} 天（本次清理过期任务 ${res.cleaned.deleted_count} 个）`,
      });
    } catch (err: any) {
      setNotice({ type: "err", text: err.message });
    }
  };

  const handleSaveCacheDir = async (targetPath: string) => {
    setNotice(null);
    try {
      setSavingCacheDir(true);
      const res = await api.adminSetCacheDir(targetPath.trim());
      setCacheDirInput(res.storage.cache_dir);
      setCacheDirTouched(false);
      await fetchOverview();
      setNotice({
        type: "ok",
        text: res.storage.is_custom_cache_dir
          ? `缓存存储路径已切换并立即生效：${res.storage.cache_dir}（可用空间 ${res.storage.disk_free_gb} GB）`
          : `已恢复默认本地缓存路径：${res.storage.cache_dir}`,
      });
    } catch (err: any) {
      setNotice({ type: "err", text: err.message });
    } finally {
      setSavingCacheDir(false);
    }
  };

  if (loading) {
    return (
      <div className="app-shell is-centered">
        <span className="loading-note">正在加载后台状态...</span>
      </div>
    );
  }

  if (!initialized) {
    return (
      <div className="app-shell is-centered">
        <form onSubmit={handleFirstSetup} className="step-card is-form" style={{ width: 440, maxWidth: "100%" }}>
          <span className="brand-title is-large">入镜 · ShotSpace 首次启动设置</span>
          <p className="page-note">
            首次启动请设置默认监听端口、缓存存储目录与 <code>/admin</code> 后台密码。
          </p>
          {authErr && <div className="alert alert-error">{authErr}</div>}
          <div className="field">
            <label htmlFor="setup-port">默认启动端口</label>
            <input
              id="setup-port"
              type="number"
              min={1}
              max={65535}
              value={setupPort}
              onChange={(e) => setSetupPort(Number(e.target.value))}
              required
            />
          </div>
          <div className="field">
            <label htmlFor="setup-cache">缓存与任务存储路径（可选）</label>
            <input
              id="setup-cache"
              type="text"
              value={setupCacheDir}
              onChange={(e) => setSetupCacheDir(e.target.value)}
              placeholder={defaultCacheDir || "/mnt/nas/shotspace_cache（留空使用本地默认）"}
            />
          </div>
          <div className="field">
            <label htmlFor="setup-pwd">设置后台密码</label>
            <input
              id="setup-pwd"
              type="password"
              value={setupPwd}
              onChange={(e) => setSetupPwd(e.target.value)}
              placeholder="至少 4 位密码"
              required
            />
          </div>
          <div className="field">
            <label htmlFor="setup-pwd2">确认后台密码</label>
            <input
              id="setup-pwd2"
              type="password"
              value={setupPwd2}
              onChange={(e) => setSetupPwd2(e.target.value)}
              placeholder="再次输入密码"
              required
            />
          </div>
          <button type="submit" className="btn btn-lg btn-primary btn-block">
            保存并进入后台
          </button>
        </form>
      </div>
    );
  }

  if (!authed) {
    return (
      <div className="app-shell is-centered">
        <form onSubmit={handleLogin} className="step-card is-form" style={{ width: 400, maxWidth: "100%" }}>
          <div className="row-between">
            <span className="brand-title is-large">入镜 · ShotSpace 后台登录</span>
            <button type="button" className="btn btn-sm btn-ghost" onClick={onNavigateWorkbench}>
              返回工作台
            </button>
          </div>
          <p className="page-note">请输入后台管理密码。</p>
          {authErr && <div className="alert alert-error">{authErr}</div>}
          <input
            type="password"
            className="is-large"
            aria-label="后台密码"
            value={loginPwd}
            onChange={(e) => setLoginPwd(e.target.value)}
            placeholder="后台密码"
            autoFocus
            required
          />
          <button type="submit" className="btn btn-lg btn-primary btn-block">
            登录后台
          </button>
        </form>
      </div>
    );
  }

  const sum = overview?.summary;
  const activeAndQueued: QueueJobInfo[] = [
    ...(overview?.running || []),
    ...(overview?.queued || []),
  ];

  return (
    <div className="app-shell is-scroll">
      <header className="topbar">
        <div className="brand">
          <span className="brand-title">入镜 · ShotSpace 后台管理</span>
          {appVersion && (
            <span className="brand-version" title="当前运行的版本">
              v{appVersion}
            </span>
          )}
          <span className="brand-sub">
            可用显卡 {sum?.enabled_gpu_count ?? 0}/{sum?.total_gpu_count ?? 0} · 排队{" "}
            {sum?.queued_count ?? 0} · 运行 {sum?.running_count ?? 0}
          </span>
        </div>
        <div className="topbar-right">
          <button type="button" className="btn btn-sm" onClick={onNavigateWorkbench}>
            返回工作台
          </button>
          <button
            type="button"
            className="btn btn-sm btn-ghost"
            onClick={() => {
              setAdminToken("");
              setAuthed(false);
            }}
          >
            退出登录
          </button>
        </div>
      </header>

      <div className="admin-wrap">
        {notice && (
          <div className={`alert ${notice.type === "ok" ? "alert-ok" : "alert-error"}`}>
            <div className="alert-row">
              <span>{notice.text}</span>
              <button type="button" className="btn btn-xs" onClick={() => setNotice(null)}>
                关闭
              </button>
            </div>
          </div>
        )}

        {sum?.all_gpus_disabled && (
          <div className="alert alert-warn is-large">
            <strong>全部显卡已停用：</strong>
            新任务将排队等待，启用任一显卡后自动执行。
          </div>
        )}

        {/* Multi-GPU Pool Control */}
        <section className="step-card is-roomy">
          <div className="step-header">
            <div className="step-title is-section">
              <span>显卡调度</span>
            </div>
            <div className="step-actions">
              <button
                type="button"
                className="btn btn-sm"
                onClick={async () => {
                  const data = await api.adminToggleAllGpus(true);
                  setOverview(data);
                }}
              >
                全部启用
              </button>
              <button
                type="button"
                className="btn btn-sm btn-warn"
                onClick={async () => {
                  const data = await api.adminToggleAllGpus(false);
                  setOverview(data);
                }}
              >
                全部停用
              </button>
            </div>
          </div>

          <div className="gpu-grid">
            {(overview?.gpus || []).map((g) => {
              const memPct =
                g.memory_total_mb > 0
                  ? Math.min(100, Math.round((g.memory_used_mb / g.memory_total_mb) * 100))
                  : 0;
              return (
                <div key={g.index} className={`gpu-card ${g.enabled ? "" : "is-off"}`}>
                  <div className="row-between">
                    <div className="grow">
                      <div className="gpu-card-name ellipsis" title={g.name}>
                        GPU {g.index}: {g.name}
                      </div>
                      <div className="gpu-card-sub">
                        CC {g.compute_cap} ({g.sm}) · {g.temperature_c}°C · 占用 {g.utilization_pct}%
                      </div>
                    </div>
                    <div className="row no-shrink">
                      <span
                        className={`tag ${
                          g.state === "busy" ? "tag-info" : g.enabled ? "tag-ok" : "tag-warn"
                        }`}
                      >
                        {g.state === "busy" ? "运行中" : g.enabled ? "已启用" : "已停用"}
                      </span>
                      <button
                        type="button"
                        className={`btn btn-sm ${g.enabled ? "" : "btn-primary"}`}
                        disabled={!g.compatible}
                        onClick={async () => {
                          const data = await api.adminToggleGpu(g.index, !g.enabled);
                          setOverview(data);
                        }}
                      >
                        {g.enabled ? "停用" : "启用"}
                      </button>
                    </div>
                  </div>

                  <div>
                    <div className="meter-head">
                      <span>显存占用</span>
                      <span>
                        {g.memory_used_mb} / {g.memory_total_mb} MB ({memPct}%)
                      </span>
                    </div>
                    <div className="progress is-thick">
                      <div className={`progress-fill ${g.enabled ? "" : "is-off"}`} style={{ width: `${memPct}%` }} />
                    </div>
                  </div>

                  {g.current_job ? (
                    <div className="tnum" style={{ color: "var(--accent-text)" }}>
                      正在执行：<strong>{g.current_job.shot_name}</strong> ({g.current_job.done}/
                      {g.current_job.total})
                    </div>
                  ) : (
                    <div className="dim">{g.enabled ? "空闲" : "已停用，不接收任务"}</div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Active & Queued Jobs */}
        <section className="step-card is-roomy">
          <div className="step-header">
            <div className="step-title is-section">
              <span>任务队列 ({activeAndQueued.length})</span>
            </div>
            <span className="muted" style={{ fontSize: 12 }}>
              自动刷新
            </span>
          </div>

          {activeAndQueued.length === 0 ? (
            <div className="meta-pill is-empty">当前队列空闲</div>
          ) : (
            <div className="col gap-6">
              {activeAndQueued.map((j) => (
                <div key={j.id} className="job-row">
                  <div className="job-row-main">
                    <span className={`tag ${j.status === "running" ? "tag-info" : "tag-warn"}`}>
                      {j.status === "running"
                        ? `GPU ${j.gpu_index} 运行中`
                        : `排队 #${j.queue_position}`}
                    </span>
                    <div>
                      <strong>{j.shot_name}</strong>{" "}
                      <span className="dim">
                        ({j.job_type === "solve" ? "三维解算" : "人物检测"} · {j.client_ip || "本机"})
                      </span>
                      <div className="job-row-msg" title={j.message}>
                        {j.message}
                      </div>
                    </div>
                  </div>

                  {j.status === "running" && j.total > 0 && (
                    <div className="job-row-meter">
                      <div className="progress is-thick">
                        <div
                          className="progress-fill"
                          style={{ width: `${Math.min(100, Math.round((j.done / j.total) * 100))}%` }}
                        />
                      </div>
                      <div className="job-row-count">
                        {j.done}/{j.total} 帧
                      </div>
                    </div>
                  )}

                  <button
                    type="button"
                    className="btn btn-sm btn-danger"
                    onClick={async () => {
                      const res = await api.adminCancelJob(j.id);
                      setOverview(res.overview);
                    }}
                  >
                    {j.status === "running" ? "终止" : "取消"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* v0.2.12: whose task is whose - every computer and its tasks; the admin gives a task to a computer,
            sets it unclassified, or deletes it */}
        <section className="step-card is-roomy">
          <div className="step-title is-section">
            <span>任务归属（{taskOwners.length} 个任务）</span>
            <input
              type="text"
              placeholder="搜索任务或电脑..."
              aria-label="搜索任务或电脑"
              value={ownerSearch}
              onChange={(e) => setOwnerSearch(e.target.value)}
              style={{ width: 200 }}
            />
          </div>
          <div className="hint">
            每个任务只有建它的电脑能看到。「未分类」是升级前建的、未能确定属于哪台电脑的任务，所有电脑都能看到：
            可以在这里分给对应的电脑，或删除。分错了电脑的任务，也在这里重新分配。点一台电脑只看它的任务。
          </div>
          <div className="owner-machines">
            {machines.map((m) => {
              const k = m.unclassified ? "__unclassified" : m.key;
              const isEmptyMachine = !m.unclassified && m.task_count === 0;
              return (
                <button
                  key={k || m.label}
                  type="button"
                  className={`owner-chip ${ownerFilter === k ? "active" : ""} ${
                    m.unclassified && m.task_count > 0 ? "is-warn" : ""
                  } ${isEmptyMachine ? "is-empty" : ""}`}
                  aria-pressed={ownerFilter === k}
                  onClick={() => setOwnerFilter((prev) => (prev === k ? null : k))}
                >
                  <strong>{m.label}</strong>
                  <span className="tnum">{m.task_count} 个任务</span>
                  {m.last_seen && <span className="dim">最近打开 {shortTime(m.last_seen)}</span>}
                  {isEmptyMachine && <span className="dim">打开过页面，没有任务</span>}
                </button>
              );
            })}
          </div>
          {shownOwners.length === 0 ? (
            <div className="meta-pill is-empty">{taskOwners.length === 0 ? "暂无任务" : "没有符合条件的任务"}</div>
          ) : (
            <div className="history-list">
              {shownOwners.map((t) => {
                const busyRow = ownerBusy === t.id;
                const running = t.status === "queued" || t.status === "running";
                return (
                  <div key={t.id} className="job-row owner-row">
                    <div className="grow">
                      <span className={`tag ${t.is_public ? "tag-warn" : "tag-muted"}`} style={{ marginRight: 8, verticalAlign: "middle" }}>
                        {t.owner_label}
                      </span>
                      <strong>{t.name}</strong>{" "}
                      <span className="dim">
                        ({t.people_count > 0 ? `已解算 ${t.people_count} 人` : "未解算"}
                        {running ? " · 计算中" : ""}
                        {t.owner_how ? ` · ${t.owner_how}` : ""}
                        {t.owner_why ? ` · ${t.owner_why}` : ""})
                      </span>
                    </div>
                    <div className="owner-actions">
                      {assignOtherFor === t.id ? (
                        <>
                          <input
                            type="text"
                            placeholder="电脑的 IP 地址"
                            aria-label="电脑的 IP 地址"
                            value={assignOtherIp}
                            onChange={(e) => setAssignOtherIp(e.target.value)}
                            style={{ width: 140 }}
                            autoFocus
                          />
                          <button
                            type="button"
                            className="btn btn-xs btn-primary"
                            disabled={busyRow || !assignOtherIp.trim()}
                            onClick={() => runOwnerAction(t.id, () => api.adminAssignTask(t.id, assignOtherIp.trim(), ""))}
                          >
                            分配
                          </button>
                          <button type="button" className="btn btn-xs" onClick={() => setAssignOtherFor(null)}>
                            取消
                          </button>
                        </>
                      ) : (
                        <select
                          value=""
                          aria-label={`把「${t.name}」分配给一台电脑`}
                          disabled={busyRow}
                          onChange={(e) => {
                            const v = e.target.value;
                            if (!v) return;
                            if (v === "__other") {
                              setAssignOtherIp("");
                              setAssignOtherFor(t.id);
                              return;
                            }
                            const m = machines.find((x) => !x.unclassified && x.key === v);
                            if (m) runOwnerAction(t.id, () => api.adminAssignTask(t.id, m.ip, m.name));
                          }}
                        >
                          <option value="">分配给…</option>
                          {machines
                            .filter((m) => !m.unclassified && m.key !== t.owner_key)
                            .map((m) => (
                              <option key={m.key} value={m.key}>
                                {m.label}
                              </option>
                            ))}
                          <option value="__other">其他电脑（输入 IP）…</option>
                        </select>
                      )}
                      {!t.is_public && (
                        <button
                          type="button"
                          className="btn btn-xs"
                          disabled={busyRow}
                          onClick={() => runOwnerAction(t.id, () => api.adminMakeTaskUnclassified(t.id))}
                        >
                          设为未分类
                        </button>
                      )}
                      {confirmDeleteTask === t.id ? (
                        <button
                          type="button"
                          className="btn btn-xs btn-danger is-confirm"
                          disabled={busyRow}
                          onClick={() => runOwnerAction(t.id, () => api.adminDeleteTask(t.id))}
                          onMouseLeave={() => setConfirmDeleteTask(null)}
                        >
                          {running ? "确认删除（会取消计算）" : "确认删除"}
                        </button>
                      ) : (
                        <button type="button" className="btn btn-xs btn-danger" disabled={busyRow} onClick={() => setConfirmDeleteTask(t.id)}>
                          删除
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* History + Data Retention & Security */}
        <div className="admin-grid">
          <section className="step-card is-roomy">
            <div className="step-title is-section">
              <span>历史记录</span>
            </div>
            {(overview?.history || []).length === 0 ? (
              <div className="meta-pill is-empty">暂无历史记录</div>
            ) : (
              <div className="history-list">
                {(overview?.history || []).slice(0, 25).map((h) => (
                  <div key={h.id} className="job-row">
                    <div className="grow">
                      <span
                        className={`tag ${
                          h.status === "completed"
                            ? "tag-ok"
                            : h.status === "cancelled"
                            ? "tag-warn"
                            : "tag-err"
                        }`}
                        style={{ marginRight: 8, verticalAlign: "middle" }}
                      >
                        {h.status === "completed"
                          ? "已完成"
                          : h.status === "cancelled"
                          ? "已取消"
                          : "失败"}
                      </span>
                      <strong>{h.shot_name}</strong>{" "}
                      <span className="dim">
                        ({h.job_type === "solve" ? "三维解算" : "人物检测"}
                        {h.gpu_index !== null ? ` · GPU ${h.gpu_index}` : ""})
                      </span>
                      {h.error && <div className="job-row-err">{h.error}</div>}
                    </div>
                    <span className="dim tnum nowrap" style={{ fontSize: 11 }}>
                      {h.duration_sec != null ? `${h.duration_sec}s` : ""}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="step-card is-roomy" style={{ gap: 14 }}>
            <div className="step-title is-section">
              <span>缓存存储路径、保留期限与安全设置</span>
            </div>

            <div className="settings-block">
              <div className="row-between">
                <span className="settings-title">缓存与任务存储路径</span>
                {overview?.server && (
                  <span
                    className={`tag ${overview.server.is_custom_cache_dir ? "tag-info" : "tag-ok"}`}
                  >
                    {overview.server.is_custom_cache_dir ? "自定义服务器路径" : "本地默认路径"}
                  </span>
                )}
              </div>
              <div className="meta-pill" style={{ wordBreak: "break-all" }}>
                <div>
                  当前生效：<strong>{overview?.server?.cache_dir || "—"}</strong>
                </div>
                <div>
                  磁盘可用 {overview?.server?.disk_free_gb ?? 0} GB / 总计{" "}
                  {overview?.server?.disk_total_gb ?? 0} GB · 当前缓存占用{" "}
                  {overview?.server?.total_mb ?? 0} MB（共 {overview?.server?.shots_total ?? 0} 个任务）
                </div>
                {overview?.server?.data_collect_dir && (
                  <div>
                    数据收集表（每天一张）：<strong>{overview.server.data_collect_dir}</strong> · 今日{" "}
                    {overview.server.data_collect_today_mb ?? 0} MB · 累计 {overview.server.data_collect_total_mb ?? 0} MB
                  </div>
                )}
              </div>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSaveCacheDir(cacheDirInput);
                }}
                className="col"
              >
                <input
                  type="text"
                  aria-label="缓存与任务存储路径"
                  value={cacheDirInput}
                  onChange={(e) => {
                    setCacheDirTouched(true);
                    setCacheDirInput(e.target.value);
                  }}
                  placeholder="服务器绝对路径，如 /mnt/nas/shotspace_cache"
                  style={{ width: "100%" }}
                />
                <div className="row">
                  <button
                    type="submit"
                    className={`btn btn-grow ${savingCacheDir ? "is-busy" : ""}`}
                    disabled={savingCacheDir}
                  >
                    {savingCacheDir && <span className="spinner" />}
                    {savingCacheDir ? "正在切换..." : "保存并切换缓存路径"}
                  </button>
                  {overview?.server?.is_custom_cache_dir && (
                    <button
                      type="button"
                      className="btn"
                      disabled={savingCacheDir}
                      onClick={() => handleSaveCacheDir("")}
                    >
                      恢复默认路径
                    </button>
                  )}
                </div>
              </form>
            </div>

            <div className="settings-block">
              <div className="row-between">
                <span className="settings-title">解算数据保留天数（默认 30 天）</span>
                <span className="muted tnum" style={{ fontSize: 11 }}>
                  {overview?.server?.solved_shots ?? 0} 个已解算任务
                </span>
              </div>
              <div className="row row-wrap gap-6">
                {[7, 15, 30, 90, 0].map((d) => (
                  <button
                    key={d}
                    type="button"
                    aria-pressed={retentionDays === d}
                    className={`btn btn-sm ${retentionDays === d ? "is-on" : ""}`}
                    onClick={() => {
                      setRetentionDays(d);
                      handleSaveRetention(d);
                    }}
                  >
                    {d === 0 ? "永久保留" : d === 30 ? "30 天 (默认)" : `${d} 天`}
                  </button>
                ))}
              </div>
              <div className="row">
                <input
                  type="number"
                  aria-label="解算数据保留天数"
                  min={0}
                  max={3650}
                  value={retentionDays}
                  onChange={(e) => {
                    setRetentionTouched(true);
                    setRetentionDays(Number(e.target.value));
                  }}
                  style={{ flex: 1, minWidth: 0 }}
                />
                <button
                  type="button"
                  className="btn"
                  onClick={() => handleSaveRetention(retentionDays)}
                >
                  保存设置
                </button>
              </div>
            </div>

            <div className="settings-block">
              <form onSubmit={handleChangePassword} className="col">
                <div className="settings-title">修改后台密码</div>
                <input
                  type="password"
                  placeholder="当前密码"
                  value={oldPwd}
                  onChange={(e) => setOldPwd(e.target.value)}
                  required
                />
                <input
                  type="password"
                  placeholder="新密码（至少 4 位）"
                  value={newPwd}
                  onChange={(e) => setNewPwd(e.target.value)}
                  required
                />
                <input
                  type="password"
                  placeholder="确认新密码"
                  value={newPwd2}
                  onChange={(e) => setNewPwd2(e.target.value)}
                  required
                />
                <button type="submit" className="btn">
                  修改密码
                </button>
              </form>
            </div>

            <div className="settings-block">
              <form onSubmit={handleSavePort} className="col">
                <div className="settings-title">默认启动端口</div>
                <div className="row">
                  <input
                    type="number"
                    aria-label="默认启动端口"
                    min={1}
                    max={65535}
                    value={cfgPort}
                    onChange={(e) => {
                      setPortTouched(true);
                      setCfgPort(Number(e.target.value));
                    }}
                    style={{ flex: 1, minWidth: 0 }}
                  />
                  <button type="submit" className="btn">
                    保存端口
                  </button>
                </div>
              </form>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
