"""Shared implementation primitives for BCPR.

PROBLEM. For a current query q_t and a candidate track k, the stored observations
h_{k,i} of that SAME track can differ in how much they help separate k from the
competing identities. The quantity we want is relational and query-conditioned:

    U(q_t, h_{k,i} ; H_k, H_competitors)

STRUCTURAL DIAGNOSIS OF THE PREVIOUS METHOD (CDE). CDE computed two scalars,
    c = <q_hat, h_hat_i>                      (query compatibility)
    g = sum_j alpha_j max_m <h_hat_i, h_hat_jm>  (competitor genericness)
and fed MLP(c, g, ...). The target is a function of 256-D vectors, but CDE collapses
the relation into two inner products BEFORE any parameter is learned. An MLP on two
scalars cannot recover relational structure that the projection to scalars destroyed.
This is a representational-capacity argument about the architecture; it does not depend
on any statistic of the frozen MOTIP features.

WHAT THE CROSS-DOMAIN LITERATURE SAYS TO DO INSTEAD.
  End-To-End Memory Networks (NeurIPS 2015): importance is NOT stored. It is decided at
    READ time, by a query-conditioned softmax over inner products in LEARNED embedding
    spaces, and the read output is a VECTOR that is added back to the query state.
  Cutie (CVPR 2024): top-down, query-based object-level reading is more robust to
    distractors than bottom-up similarity matching.
  DAM4SAM (CVPR 2025): keep distractor-resolving evidence as its own pathway rather than
    folding it into one appearance score.
  k-reciprocal re-ranking (CVPR 2017): the relation among the candidates corrects a
    pairwise similarity that is not discriminative on its own.

BCPR reads competitor context as a vector in a query-conditioned learned subspace
and form a per-observation RELATIONAL vector before any scalar is taken:

    P        : learned 256 -> d projection, shared by query and history (MemN2N A/C)
    o_comp(q): MemN2N read over the COMPETING tracks' observations only
    r_i      : elementwise interactions of (Pq, Ph_i, o_comp) + track/temporal context
    u_i      : MLP(r_i), zero-initialised output
    bias     : row-centred lambda*tanh(u_i) added to the history cross-attention logits

Cost. The competitor context is summarised FROM THE QUERY, so BCPR needs only
query<->history reads: O(Q*K*d). CDE needed history<->history similarity: O(K^2*d).
At inference Q << K, so this removes the dominant term of the previous method.

Preserved from the verified CDE implementation (all covered by regression tests):
exact no-op at init, row-centering over valid causal keys, padding/causal masks
authoritative, eps inside every sqrt, torch.where masking instead of sentinel values,
query-chunked + checkpointed body, RNG-neutral construction, detached diagnostics.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint as cp
from torch import nn
import torch.nn.functional as F


def _row_standardise(v, valid, eps=1e-6):
    """Standardise over each query row's valid keys. eps INSIDE the sqrt: with
    var.sqrt() + eps the forward is finite but d sqrt/d var is inf at var = 0, which a
    fully-masked row reaches, and the following multiply by the zero mask makes inf*0 =
    NaN in the backward pass."""
    m = valid.to(v.dtype)
    n = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (v * m).sum(dim=-1, keepdim=True) / n
    var = (((v - mean) ** 2) * m).sum(dim=-1, keepdim=True) / n
    return ((v - mean) * (var + eps).rsqrt()) * m


class BCPRBase(nn.Module):
    """Base implementation used by the public :class:`BCPR` module."""

    def __init__(self, feature_dim: int = 256, proj_dim: int = 64, hidden_dim: int = 64,
                 lam: float = 1.0, tau_init: float = 1.0, time_scale: float = 30.0,
                 use_competitor: bool = True, chunk_queries: int = 128,
                 shuffle_control: bool = False, use_qh: bool = True,
                 use_hc: bool = True, use_qc: bool = True,
                 use_own: bool = True):
        super().__init__()
        # Building this module must not perturb the global RNG stream: PyTorch draws the
        # DataLoader workers' base_seed from the DEFAULT generator when the loader's
        # iterator is created, which happens AFTER the model is built. Consuming draws
        # here silently changes the augmentation an arm sees relative to its baseline
        # (measured previously: identical weights, different subsequent draws).
        _rng = torch.get_rng_state()
        self.enabled = True
        self.lam = float(lam)
        self.time_scale = float(time_scale)
        self.shuffle_control = bool(shuffle_control)
        self.use_competitor = bool(use_competitor)
        # Factorial ablation switches for the three 64-D relational groups.  A
        # disabled group is zeroed rather than removed so every arm retains the
        # same parameter shapes, optimiser structure, and compute graph contract.
        self.use_qh = bool(use_qh)
        self.use_hc = bool(use_hc)
        self.use_qc = bool(use_qc)
        # Concept-level ablation switch for the own-track evidence branch.
        # Validity remains an authoritative mask rather than an ablatable cue;
        # OWN controls only causal track consistency (rho) and temporal gap.
        self.use_own = bool(use_own)
        self.chunk_queries = int(chunk_queries)
        d = int(proj_dim)
        self.d = d
        # Shared key/query projection (MemN2N's input embedding A) and a separate value
        # projection for the read output (MemN2N's output embedding C). Separating them
        # is what lets "what makes this competitor threatening" differ from "what the
        # competitor contributes to the context vector".
        self.proj = nn.Linear(feature_dim, d, bias=False)
        self.val = nn.Linear(feature_dim, d, bias=False)
        # Read temperature. tau multiplies a cosine in [-1, 1] after a learned
        # projection, so it is parameterised in log space and initialised at 1.0 rather
        # than at the previous method's hand-picked 0.1: the projection is free to set
        # its own scale, so there is no reason to pre-sharpen the softmax.
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())
        # r_i = [Pq * Ph_i, Ph_i - o_comp, Pq * o_comp, rho, dt, valid]
        self.n_in = 3 * d + 3
        self.mlp = nn.Sequential(
            nn.Linear(self.n_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_stats = {}
        torch.set_rng_state(_rng)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _unit(x, eps: float = 1e-12):
        return x * (x.pow(2).sum(dim=-1, keepdim=True) + eps).rsqrt()

    def forward(self, q_feat, h_feat, valid, track_id, times, query_time, causal):
        """
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
        if not self.enabled or K == 0:
            return q_feat.new_zeros((BG, Q, K))
        vis = causal & valid.unsqueeze(1)
        n_tr = int(track_id.max().item()) + 1 if track_id.numel() else 0
        if n_tr <= 0:
            return q_feat.new_zeros((BG, Q, K))

        pk = self._unit(self.proj(h_feat))                     # BG,K,d  key side
        pq = self._unit(self.proj(q_feat))                     # BG,Q,d  query side
        vk = self.val(h_feat)                                  # BG,K,d  value side
        key_track = track_id.clamp_min(0)

        # rho: how the observation sits relative to its own track's direction statistic
        # in the LEARNED space. A direction statistic, not a mean of magnitudes, so a
        # single large-norm observation cannot dominate it.
        oh = F.one_hot(key_track, n_tr).to(pk.dtype) * valid.unsqueeze(-1).to(pk.dtype)
        tr_dir = self._unit(torch.einsum("bkd,bkt->btd", pk, oh))
        rho = (pk * tr_dir.gather(1, key_track.unsqueeze(-1).expand(-1, -1, self.d))).sum(-1)
        rho = (rho * valid.to(rho.dtype))                      # BG,K

        step = self.chunk_queries if self.chunk_queries > 0 else Q
        out, stat = [], []
        for q0 in range(0, Q, step):
            q1 = min(q0 + step, Q)
            args = (pq[:, q0:q1], pk, vk, vis[:, q0:q1], rho, query_time[:, q0:q1])

            def _body(pqc, pk_, vk_, visc, rho_, qt):
                # Qc from the input shape, never from the enclosing loop variable:
                # checkpoint(use_reentrant=False) re-runs this in the backward pass, by
                # which time a closed-over Qc holds the last chunk's size.
                Qc = pqc.shape[1]
                s = torch.einsum("bqd,bkd->bqk", pqc, pk_)     # BG,Qc,K  learned-space sim
                if self.use_competitor:
                    # MemN2N read restricted to the COMPETING tracks of each key's own
                    # track. Masking is boolean throughout: a finite sentinel such as
                    # torch.finfo.min passes isfinite() and overflows when divided by
                    # tau, and a multiplicative mask turns the resulting inf into NaN.
                    tau = self.log_tau.exp().clamp(1e-2, 1e2)
                    own = F.one_hot(key_track, n_tr).bool()    # BG,K,n_tr
                    # per (query, track) read weight over that track's visible keys
                    lg = (s / tau).unsqueeze(-1)               # BG,Qc,K,1
                    sel = visc.unsqueeze(-1) & (oh > 0).unsqueeze(1)   # BG,Qc,K,n_tr
                    lgm = lg.masked_fill(~sel, -1e4)
                    mx = lgm.amax(dim=2, keepdim=True)
                    any_t = sel.any(dim=2, keepdim=True)
                    mx = torch.where(any_t, mx, torch.zeros_like(mx))
                    w = torch.where(sel, (lgm - mx).clamp(min=-60.0, max=0.0).exp(),
                                    torch.zeros_like(lgm))     # BG,Qc,K,n_tr
                    # read vector per (query, track)
                    num = torch.einsum("bqkt,bkd->bqtd", w, vk_)
                    den = w.sum(dim=2)                          # BG,Qc,n_tr
                    rd = num / den.clamp_min(1e-6).unsqueeze(-1)
                    # exclude the key's own track, then pool the remaining tracks with
                    # the same query-conditioned weights so a threatening competitor
                    # contributes more to the context vector.
                    trw = den / den.sum(-1, keepdim=True).clamp_min(1e-6)   # BG,Qc,n_tr
                    trw_k = trw.unsqueeze(2).expand(-1, -1, K, -1)
                    trw_k = trw_k.masked_fill(own.unsqueeze(1), 0.0)
                    trw_k = trw_k / trw_k.sum(-1, keepdim=True).clamp_min(1e-6)
                    o_comp = torch.einsum("bqkt,bqtd->bqkd", trw_k, rd)     # BG,Qc,K,d
                else:
                    o_comp = torch.zeros(pqc.shape[0], Qc, K, self.d,
                                         dtype=pqc.dtype, device=pqc.device)
                phk = pk_.unsqueeze(1).expand(-1, Qc, -1, -1)
                pqk = pqc.unsqueeze(2).expand(-1, -1, K, -1)
                dt = ((qt.unsqueeze(-1) - times.unsqueeze(1)).clamp_min(0).to(s.dtype)
                      / self.time_scale).clamp_max(1.0)
                # Keep the competitor value/temperature path in DDP's autograd
                # graph even when both competitor-dependent feature groups are
                # ablated.  The zero anchor changes neither values nor gradients.
                comp_anchor = o_comp * 0.0
                r = torch.cat([
                    (pqk * phk if self.use_qh else torch.zeros_like(phk)) + comp_anchor,
                    phk - o_comp if self.use_hc else torch.zeros_like(phk),
                    pqk * o_comp if self.use_qc else torch.zeros_like(phk),
                    (rho.unsqueeze(1).expand(-1, Qc, -1).unsqueeze(-1)
                     if self.use_own else rho.new_zeros((rho.shape[0], Qc, K, 1))),
                    (dt.unsqueeze(-1) if self.use_own else torch.zeros_like(dt).unsqueeze(-1)),
                    visc.to(s.dtype).unsqueeze(-1),
                ], dim=-1)
                if self.shuffle_control:
                    # Attribution control (Arm D): destroy the key<->feature
                    # correspondence among VISIBLE keys only, keeping the parameter
                    # count, the compute and the gradient path identical. Any gain that
                    # survives this control is capacity or regularisation, not
                    # per-observation utility.
                    #
                    # The CUDA generator is used, which is separate from the CPU default
                    # generator that DataLoader draws worker seeds from, so this does not
                    # perturb the augmentation stream. checkpoint(use_reentrant=False)
                    # defaults to preserve_rng_state=True, so the backward recomputation
                    # replays the same permutation.
                    ar = torch.arange(K, device=r.device).expand(visc.shape)
                    dst = torch.argsort(torch.where(visc, ar, ar + K), dim=-1)
                    rnd = torch.rand(visc.shape, device=r.device)
                    src = torch.argsort(torch.where(visc, rnd, rnd + 2.0), dim=-1)
                    idx = torch.zeros_like(dst).scatter_(-1, dst, src)
                    r = r.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, r.shape[-1]))
                b = self.lam * torch.tanh(self.mlp(r).squeeze(-1)) * visc.to(s.dtype)
                return b, s

            if self.training and self.chunk_queries > 0:
                b, sc = cp.checkpoint(_body, *args, use_reentrant=False)
            else:
                b, sc = _body(*args)
            mc = vis[:, q0:q1].to(b.dtype)
            nc = mc.sum(dim=-1, keepdim=True).clamp_min(1.0)
            out.append((b - (b * mc).sum(dim=-1, keepdim=True) / nc) * mc)
            stat.append(sc.detach())
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
            }
        return b_tilde
