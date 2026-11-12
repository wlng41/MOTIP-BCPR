"""BCPR observation-wise attention refinement.

BCPR extends :class:`models.motip.bcpr_base.BCPRBase` with length-neutral
competitor threat and query-time causal own-track consistency. It overrides only
``forward``, so the module's
parameters, their names, shapes, initialisation order and the RNG-neutral
construction are inherited verbatim and cannot drift from the original.

MODIFICATION A — length-neutral absolute competitor threat.
  Original BCPR reused the within-track softmax's max-shifted denominator as the
  cross-track competitor weight:

      den_t   = sum_{i in H_t} exp((s_i - max_{H_t} s) / tau)
      trw_t   = den_t / sum_t' den_t'

  Subtracting each track's OWN maximum removes the absolute similarity, so den_t
  degenerates towards |H_t| and trw_t towards a pure observation-count share.  A
  short track can then never out-weigh a long one however well it matches the
  query (measured: singleton at s=0.9 -> 1.0, ten copies at s=0.1 -> 10.0).

  BCPR uses the length-neutral log-mean-exp threat

      e_t = tau * [ log sum_{i in H_t} exp(s_i / tau) - log |H_t| ]
      pi_t = softmax_t( e_t / tau )

  e_t keeps the absolute similarity (shifting every s_i of a track by c shifts
  e_t by c), is invariant to duplicating a track's observations, and reduces to
  e_t = s_i for a singleton track.  Because e_t / tau is exactly the log-mean-exp
  of s/tau, it is read straight off the max-shifted quantities the within-track
  readout already computes -- no extra tensor and no extra parameter.

MODIFICATION B — query-time causal rho.
  Original BCPR built each track's direction statistic from ALL valid keys, with
  no query dimension:

      tr_dir_t = unit( sum_{i: track(i)=t, valid_i} p_i )

  so rho_i carried information from observations LATER than the query.  BCPR
  builds the statistic from the causal prefix of each query:

      d_{q,t}  = unit( sum_{i: track(i)=t, visible to q} p_i )
      rho_{qi} = p_i . d_{q, track(i)}

  Training and inference are then causally consistent, and a future observation
  can influence neither the statistic nor its gradient.

The within-track value readout keeps the original stable max-shifted softmax, and
the single learned temperature is SHARED by the readout and the threat exactly as
in the original: the first BCPR run adds no new learned degree of freedom, so
the comparison isolates the two corrections above.

Everything else is inherited or copied unchanged: the shared 256->d projection,
the separate value projection, the 3d+3 relational feature, the zero-initialised
MLP output (exact no-op at init), lambda*tanh magnitude control, arithmetic
row-centering over valid causal keys, boolean masking throughout with eps inside
every sqrt, the query-chunked checkpointed body, and the detached diagnostics.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint as cp
import torch.nn.functional as F

from models.motip.bcpr_base import BCPRBase, _row_standardise


def competitor_threat_logits(similarity, selection, tau, neg=-1e4):
    """Reference (unfused) length-neutral absolute threat, in units of e/tau.

    This is the plain transcription of

        e_t / tau = log mean_{i in H_t} exp(s_i / tau)

    written independently of :meth:`BCPR.forward`, which fuses the same quantity
    onto the max-shifted ``mx``/``den`` the within-track readout already needs.  The
    preflight suite asserts the two agree, so the fused path is checked against a
    form that can be read off the definition.

    similarity  [..., K]      learned-space query/key similarity
    selection   [..., K, T]   True where key i belongs to track t AND is visible
    returns     ([..., T] log-mean-exp of s/tau, [..., T] bool non-empty)
    """
    logits = (similarity / tau).unsqueeze(-1)
    present = selection.any(dim=-2)
    masked = logits.masked_fill(~selection, neg)
    maximum = masked.amax(dim=-2)
    safe_max = torch.where(present, maximum, torch.zeros_like(maximum))
    shifted = torch.where(
        selection,
        (logits - safe_max.unsqueeze(-2)).clamp(min=-60.0, max=0.0).exp(),
        torch.zeros_like(logits),
    )
    count = selection.sum(dim=-2).clamp_min(1).to(similarity.dtype)
    lme = safe_max + shifted.sum(dim=-2).clamp_min(1e-30).log() - count.log()
    return torch.where(present, lme, torch.zeros_like(lme)), present


class BCPR(BCPRBase):
    """BCPR with length-neutral threat and query-time causal consistency."""

    # Masking sentinel.  Applied only to quantities that are ALREADY divided by
    # tau, so -- unlike a sentinel placed on the raw similarity -- it is never
    # divided again and cannot overflow.  s/tau is bounded by 1/1e-2 = 100 and
    # log-mean-exp of it by the same bound, so -1e4 is unreachable from data and
    # yields exactly-zero softmax weight, while staying finite in the backward
    # pass (-inf would make 0 * -inf = NaN in a masked product).
    _NEG = -1e4

    # Diagnostic capture of the corrected threat.  Default False: when off, forward
    # stores nothing extra and is unaffected.  Set it only in eval/diagnostic code --
    # under gradient checkpointing the backward recomputation overwrites the capture,
    # which is harmless for diagnostics but makes it useless during training.
    capture_threat = False

    def forward(self, q_feat, h_feat, valid, track_id, times, query_time, causal):
        """Signature, shapes and return value are identical to BCPR.forward.

        q_feat     [BG, Q, F]  raw current-query features
        h_feat     [BG, K, F]  raw stored history features
        valid      [BG, K]     True on real keys
        track_id   [BG, K]     track/column of each key, -1 if invalid
        times      [BG, K]     key time index
        query_time [BG, Q]     query time index
        causal     [BG, Q, K]  True where the key is visible to the query
        returns    [BG, Q, K]  row-centred bias
        """
        BG, Q, _ = q_feat.shape
        K = h_feat.shape[1]
        # Some evaluation frames contain no current detections (Q == 0).  In
        # that valid runtime state there is no bias row to compute; return the
        # shape-correct no-op instead of reaching torch.cat on an empty chunk
        # list.  K == 0 is the analogous no-history case.
        if not self.enabled or Q == 0 or K == 0:
            return q_feat.new_zeros((BG, Q, K))
        vis = causal & valid.unsqueeze(1)
        n_tr = int(track_id.max().item()) + 1 if track_id.numel() else 0
        if n_tr <= 0:
            return q_feat.new_zeros((BG, Q, K))

        pk = self._unit(self.proj(h_feat))                     # BG,K,d  key side
        pq = self._unit(self.proj(q_feat))                     # BG,Q,d  query side
        vk = self.val(h_feat)                                  # BG,K,d  value side
        key_track = track_id.clamp_min(0)
        # Track membership of each key, zeroed on padding.  rho's causal prefix is
        # now built per query inside the body, so -- unlike the original -- there is
        # no query-independent direction statistic to precompute here.
        oh_b = (F.one_hot(key_track, n_tr) * valid.unsqueeze(-1)).bool()   # BG,K,n_tr
        own = F.one_hot(key_track, n_tr).bool()                            # BG,K,n_tr

        step = self.chunk_queries if self.chunk_queries > 0 else Q
        out, stat, thr = [], [], []
        for q0 in range(0, Q, step):
            q1 = min(q0 + step, Q)
            args = (pq[:, q0:q1], pk, vk, vis[:, q0:q1], query_time[:, q0:q1])

            def _body(pqc, pk_, vk_, visc, qt):
                # Qc from the input shape, never from the enclosing loop variable:
                # checkpoint(use_reentrant=False) re-runs this in the backward pass,
                # by which time a closed-over Qc holds the last chunk's size.
                Qc = pqc.shape[1]
                s = torch.einsum("bqd,bkd->bqk", pqc, pk_)     # BG,Qc,K  learned-space sim
                vf = visc.to(s.dtype)
                tau = self.log_tau.exp().clamp(1e-2, 1e2)

                # --- MODIFICATION B: query-time causal rho -------------------
                # d_{q,t} = unit(sum over the keys of track t VISIBLE TO q).  The
                # scatter-add is over the track axis, which keeps the intermediate
                # at [BG,Qc,n_tr,d] instead of the [BG,K,n_tr,d] a direct einsum
                # over the one-hot would materialise.  Padding keys carry
                # key_track = 0 after clamp_min, but vf is zero on them, so they
                # contribute exactly nothing to track 0.
                gi = key_track[:, None, :, None].expand(-1, Qc, -1, self.d)
                ctb = pk_.unsqueeze(1) * vf.unsqueeze(-1)              # BG,Qc,K,d
                pre = ctb.new_zeros((pqc.shape[0], Qc, n_tr, self.d)).scatter_add_(2, gi, ctb)
                # _unit keeps eps INSIDE the rsqrt, so an empty prefix gives the
                # zero direction with a finite gradient rather than inf * 0 = NaN.
                own_dir = self._unit(pre).gather(2, gi)                 # BG,Qc,K,d
                rho_q = (pk_.unsqueeze(1) * own_dir).sum(-1) * vf       # BG,Qc,K

                if self.use_competitor:
                    # Within-track read weights: unchanged stable max-shifted softmax.
                    sel = visc.unsqueeze(-1) & oh_b.unsqueeze(1)       # BG,Qc,K,n_tr
                    lg = (s / tau).unsqueeze(-1)                       # BG,Qc,K,1
                    lgm = lg.masked_fill(~sel, self._NEG)
                    mx = lgm.amax(dim=2, keepdim=True)                 # BG,Qc,1,n_tr
                    any_t = sel.any(dim=2, keepdim=True)
                    mx = torch.where(any_t, mx, torch.zeros_like(mx))
                    w = torch.where(sel, (lgm - mx).clamp(min=-60.0, max=0.0).exp(),
                                    torch.zeros_like(lgm))             # BG,Qc,K,n_tr
                    num = torch.einsum("bqkt,bkd->bqtd", w, vk_)
                    den = w.sum(dim=2)                                  # BG,Qc,n_tr
                    rd = num / den.clamp_min(1e-6).unsqueeze(-1)        # BG,Qc,n_tr,d

                    # --- MODIFICATION A: length-neutral absolute threat ------
                    # e_t / tau = log mean_i exp(s_i / tau)
                    #           = mx_t + log(sum_i exp(s_i/tau - mx_t)) - log|H_t|.
                    # mx and den are exactly those two max-shifted quantities, so
                    # the corrected threat costs one log and one subtraction.  The
                    # softmax is over e_t / tau, so tau cancels and no separate
                    # temperature is introduced.
                    present = any_t.squeeze(2)                          # BG,Qc,n_tr
                    cnt = sel.sum(dim=2).clamp_min(1).to(s.dtype)       # BG,Qc,n_tr
                    lme = mx.squeeze(2) + den.clamp_min(1e-30).log() - cnt.log()
                    trl = lme.masked_fill(~present, self._NEG)
                    trw = torch.softmax(trl, dim=-1)
                    trw = torch.where(present, trw, torch.zeros_like(trw))

                    # Exclude the key's own track, then renormalise -- unchanged.
                    trw_k = trw.unsqueeze(2).expand(-1, -1, K, -1)
                    trw_k = trw_k.masked_fill(own.unsqueeze(1), 0.0)
                    trw_k = trw_k / trw_k.sum(-1, keepdim=True).clamp_min(1e-6)
                    o_comp = torch.einsum("bqkt,bqtd->bqkd", trw_k, rd)  # BG,Qc,K,d
                    if self.capture_threat:
                        self.last_threat = {
                            "similarity": s.detach(), "selection": sel.detach(),
                            "lme": lme.detach(), "present": present.detach(),
                            "trw": trw.detach(), "tau": tau.detach(),
                        }
                    thr_s = torch.stack([
                        (lme * present).sum() / present.sum().clamp_min(1),
                        lme.masked_fill(~present, 0.0).abs().max(),
                        trw.max(),
                    ]).detach()
                else:
                    o_comp = torch.zeros(pqc.shape[0], Qc, K, self.d,
                                         dtype=pqc.dtype, device=pqc.device)
                    thr_s = s.new_zeros(3).detach()

                phk = pk_.unsqueeze(1).expand(-1, Qc, -1, -1)
                pqk = pqc.unsqueeze(2).expand(-1, -1, K, -1)
                dt = ((qt.unsqueeze(-1) - times.unsqueeze(1)).clamp_min(0).to(s.dtype)
                      / self.time_scale).clamp_max(1.0)
                # Preserve a zero-valued autograd path through the competitor read
                # for factorial arms that disable both competitor-dependent groups.
                # This avoids DDP unused-parameter failures without changing outputs.
                comp_anchor = o_comp * 0.0
                r = torch.cat([
                    (pqk * phk if self.use_qh else torch.zeros_like(phk)) + comp_anchor,
                    phk - o_comp if self.use_hc else torch.zeros_like(phk),
                    pqk * o_comp if self.use_qc else torch.zeros_like(phk),
                    (rho_q.unsqueeze(-1) if self.use_own
                     else torch.zeros_like(rho_q).unsqueeze(-1)),
                    (dt.unsqueeze(-1) if self.use_own
                     else torch.zeros_like(dt).unsqueeze(-1)),
                    vf.unsqueeze(-1),
                ], dim=-1)
                if self.shuffle_control:
                    # Identical attribution control to BCPR: destroy the
                    # key<->feature correspondence among VISIBLE keys only, keeping
                    # the parameter count, the compute and the gradient path the
                    # same.  Uses the CUDA generator, which is separate from the CPU
                    # default generator DataLoader draws worker seeds from, and
                    # checkpoint(use_reentrant=False) preserves RNG state so the
                    # backward recomputation replays the same permutation.
                    ar = torch.arange(K, device=r.device).expand(visc.shape)
                    dst = torch.argsort(torch.where(visc, ar, ar + K), dim=-1)
                    rnd = torch.rand(visc.shape, device=r.device)
                    src = torch.argsort(torch.where(visc, rnd, rnd + 2.0), dim=-1)
                    idx = torch.zeros_like(dst).scatter_(-1, dst, src)
                    r = r.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, r.shape[-1]))
                b = self.lam * torch.tanh(self.mlp(r).squeeze(-1)) * vf
                return b, s, thr_s

            if self.training and self.chunk_queries > 0:
                b, sc, ts = cp.checkpoint(_body, *args, use_reentrant=False)
            else:
                b, sc, ts = _body(*args)
            mc = vis[:, q0:q1].to(b.dtype)
            nc = mc.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out.append((b - (b * mc).sum(dim=-1, keepdim=True) / nc) * mc)
            stat.append(sc.detach())
            thr.append(ts.detach())
        b_tilde = torch.cat(out, dim=1)

        with torch.no_grad():
            m = vis.to(b_tilde.dtype)
            rn = m.sum(-1)
            ok = rn > 1
            bt = b_tilde
            mu = (bt * m).sum(-1) / rn.clamp_min(1)
            var = (((bt - mu.unsqueeze(-1)) ** 2) * m).sum(-1) / rn.clamp_min(1)
            s_all = torch.cat(stat, dim=1)
            tr1 = F.one_hot(key_track, n_tr).to(bt.dtype) * valid.unsqueeze(-1).to(bt.dtype)
            cnt = torch.einsum("bqk,bkt->bqt", m, tr1)
            s1 = torch.einsum("bqk,bkt->bqt", bt * m, tr1)
            s2 = torch.einsum("bqk,bkt->bqt", (bt ** 2) * m, tr1)
            tmu = s1 / cnt.clamp_min(1)
            tvar = (s2 / cnt.clamp_min(1) - tmu ** 2).clamp_min(0)
            intr = cnt > 1
            ts_all = torch.stack(thr, dim=0)
            self.last_stats = {
                "bias_std": bt[vis].std(unbiased=False) if bool(vis.any()) else bt.sum() * 0,
                "bias_absmax": bt.abs().max(),
                "bias_row_std": var.clamp_min(0).sqrt()[ok].mean() if bool(ok.any()) else bt.sum() * 0,
                "within_track_std": tvar.sqrt()[intr].mean() if bool(intr.any()) else bt.sum() * 0,
                "sim_mean": s_all[vis].mean() if bool(vis.any()) else s_all.sum() * 0,
                "sim_row_std": _row_standardise(s_all, vis).abs().mean(),
                "tau": self.log_tau.detach().exp(),
                "proj_norm": self.proj.weight.detach().norm(),
                "val_norm": self.val.weight.detach().norm(),
                "frac_visible": m.mean(),
                # BCPR specific: the corrected threat and the competitor pooling
                # it produces.  threat_mean/absmax are in units of e/tau, and
                # trw_max is the largest single-track competitor weight, which is
                # 1.0 exactly when only one competing track is visible.
                "cf_threat_mean": ts_all[:, 0].mean(),
                "cf_threat_absmax": ts_all[:, 1].max(),
                "cf_trw_max": ts_all[:, 2].max(),
            }
        return b_tilde
