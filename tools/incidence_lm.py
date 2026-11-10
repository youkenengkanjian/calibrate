# tools/incidence_lm.py
import torch, torch.nn as nn, torch.nn.functional as F
from tools.lm_optimizer import huber_loss, scaled_loss, optimizer_step  # 直接复用

class IncidenceLM(nn.Module):
    def __init__(self, num_steps=30, lambda_=0.1, huber_scale=1e-2, use_log_f=True, min_conf=1e-6):
        super().__init__()
        self.num_steps = num_steps
        self.lambda_ = lambda_
        self.huber_scale = huber_scale
        self.use_log_f = use_log_f
        self.min_conf = min_conf

    def forward(self, incidence, confidence=None, init_K=None, aspect_ratio=None, return_cov=False):
        """
        incidence : (B,3,H,W) 单位入射向量
        confidence: (B,1,H,W) or None；若提供则作为像素权重
        init_K    : (B,3,3) or None；初值（可选）
        aspect_ratio: float or None；若提供，fy = fx * aspect_ratio（1DoF）
        return_cov: 是否返回近似协方差 H^{-1}
        """

        def _to_chw(x, want_c):
            if x is None:
                return None
            # 去掉末尾或中间的 size=1 维（常见是 [B,C,H,W,1] 或 [B,1,C,H,W]）
            while x.dim() > 4 and x.size(-1) == 1:
                x = x.squeeze(-1)
            if x.dim() == 5 and x.size(1) == 1 and x.size(2) == want_c:
                x = x.squeeze(1)
            # 若是 HWC → CHW
            if x.dim() == 4 and x.shape[1] != want_c and x.shape[-1] == want_c:
                x = x.permute(0, 3, 1, 2).contiguous()
            # 通道数不对则裁成前 want_c 个
            if x.dim() == 4 and x.shape[1] != want_c:
                x = x[:, :want_c, ...].contiguous()
            return x.contiguous()

        incidence = _to_chw(incidence, want_c=3)
        confidence = _to_chw(confidence, want_c=1)

        # 基本断言（早点报错也比在 permute 崩更友好）
        if incidence is None or incidence.dim() != 4 or incidence.shape[1] != 3:
            raise ValueError(
                f"[IncidenceLM] incidence expects [B,3,H,W], got {tuple(incidence.shape) if incidence is not None else None}")
        if confidence is not None:
            if confidence.dim() != 4:
                raise ValueError(f"[IncidenceLM] confidence expects 4D, got {tuple(confidence.shape)}")
            if confidence.shape[1] != 1:
                confidence = confidence[:, :1, ...].contiguous()

        # 保障 dtype/contiguous
        incidence = incidence.float().contiguous()
        if confidence is not None:
            confidence = confidence.float().contiguous()



        B, _, H, W = incidence.shape
        device = incidence.device

        # --------- 初值 ----------
        if init_K is None:
            fx0 = fy0 = 0.7 * float(max(H, W))
            cx0, cy0 = W / 2.0, H / 2.0
            fx = incidence.new_full((B,), fx0);
            fy = incidence.new_full((B,), fy0)
            cx = incidence.new_full((B,), cx0);
            cy = incidence.new_full((B,), cy0)
        else:
            fx = init_K[:, 0, 0].clone();
            fy = init_K[:, 1, 1].clone()
            cx = init_K[:, 0, 2].clone();
            cy = init_K[:, 1, 2].clone()

        ax = fx.log() if self.use_log_f else fx
        ay = fy.log() if self.use_log_f else fy
        ax = ax.requires_grad_(True);
        ay = ay.requires_grad_(True)
        cx = cx.requires_grad_(True);
        cy = cy.requires_grad_(True)

        # --------- 网格 ----------
        xs = torch.arange(W, device=device).float().view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.arange(H, device=device).float().view(1, 1, H, 1).expand(B, 1, H, W)

        # --------- 像素置信度 ----------
        if confidence is None:
            w = incidence.new_ones(B, 1, H, W)
        else:
            w = confidence.clamp_min(self.min_conf)  # 不 detach

        lamb = incidence.new_full((B,), self.lambda_)

        last_cost = None
        gamma = 2.0  # 置信度指数，可调

        for _ in range(self.num_steps):
            fx = ax.exp() if self.use_log_f else ax
            if aspect_ratio is None:
                fy = ay.exp() if self.use_log_f else ay
            else:
                # 1DoF：fy 由 fx 和纵横比确定
                fy = fx * float(aspect_ratio)

            # ---------- 前向：K^{-1}x → 单位入射向量 r_th ----------
            u = (xs - cx.view(B, 1, 1, 1)) / fx.view(B, 1, 1, 1)
            v = (ys - cy.view(B, 1, 1, 1)) / fy.view(B, 1, 1, 1)
            s = torch.sqrt(u * u + v * v + 1.0)
            r_th = torch.stack([u / s, v / s, 1.0 / s], dim=1)  # (B,3,H,W)

            # ---------- 残差与Huber ----------
            res = (incidence - r_th).permute(0, 2, 3, 1).reshape(B, -1, 3)  # (B,N,3)
            sq = (res * res).sum(-1)  # (B,N)
            _, w_huber, _ = scaled_loss(sq, huber_loss, self.huber_scale)
            w_tot = (w.view(B, -1).pow(gamma) * w_huber).clamp_min(1e-8)  # (B,N)

            # ---------- 解析 Jacobian ----------
            du_dfx = -u / fx.view(B, 1, 1, 1);
            du_dcx = -1.0 / fx.view(B, 1, 1, 1)
            dv_dfy = -v / fy.view(B, 1, 1, 1);
            dv_dcy = -1.0 / fy.view(B, 1, 1, 1)
            s3 = s ** 3
            dr_du = torch.stack([1 / s - (u * u) / s3, -(u * v) / s3, -(u * 1.0) / s3], dim=1)
            dr_dv = torch.stack([-(v * u) / s3, 1 / s - (v * v) / s3, -(v * 1.0) / s3], dim=1)

            J_fx = (dr_du * du_dfx).permute(0, 2, 3, 1).reshape(B, -1, 3)  # (B,N,3)
            J_fy = (dr_dv * dv_dfy).permute(0, 2, 3, 1).reshape(B, -1, 3)
            J_cx = (dr_du * du_dcx).permute(0, 2, 3, 1).reshape(B, -1, 3)
            J_cy = (dr_dv * dv_dcy).permute(0, 2, 3, 1).reshape(B, -1, 3)

            if self.use_log_f:
                J_fx = J_fx * fx.view(B, 1, 1);
                if aspect_ratio is None:
                    J_fy = J_fy * fy.view(B, 1, 1)  # 4DoF 情况
                # 1DoF 情况下，不直接使用 J_fy，见下

            # ---------- 参数维度拼装 ----------
            if aspect_ratio is None:
                # 4DoF：θ=[ax, ay, cx, cy] 或 [fx, fy, cx, cy]
                J = torch.stack([J_fx, J_fy, J_cx, J_cy], dim=-1)  # (B,N,3,4)
            else:
                # 1DoF：fy = fx * r；把 fy 的链式贡献并到 fx 的列
                if self.use_log_f:
                    # d fy / d (log fx) = fy
                    J_fx_eff = J_fx + J_fy * fy.view(B, 1, 1)
                else:
                    # d fy / d fx = r
                    J_fx_eff = J_fx + J_fy * float(aspect_ratio)
                J = torch.stack([J_fx_eff, J_cx, J_cy], dim=-1)  # (B,N,3,3)

            # ---------- 加权正规方程（关键修正：权重进 einsum） ----------
            # G = J^T W r；H = J^T W J
            G = torch.einsum('bncp,bnc,bn->bp', J, res, w_tot)  # (...,P)
            H = torch.einsum('bncp,bncq,bn->bpq', J, J, w_tot)  # (...,P,P)

            # ---------- LM 更新 ----------
            delta = optimizer_step(G, H, lamb)
            if aspect_ratio is None:
                ax = ax + delta[:, 0];
                ay = ay + delta[:, 1]
                cx = cx + delta[:, 2];
                cy = cy + delta[:, 3]
            else:
                ax = ax + delta[:, 0]  # 只有 fx（或 log fx）这一个焦距自由度
                cx = cx + delta[:, 1];
                cy = cy + delta[:, 2]

            # （可选）简单代价监控与阻尼调度，可复用 geocalib 的 update_lambda / early_stop

        fx = ax.exp() if self.use_log_f else ax
        if aspect_ratio is None:
            fy = ay.exp() if self.use_log_f else ay
        else:
            fy = fx * float(aspect_ratio)

        K = incidence.new_zeros(B, 3, 3)
        K[:, 0, 0], K[:, 1, 1] = fx, fy
        K[:, 0, 2], K[:, 1, 2] = cx, cy
        K[:, 2, 2] = 1.0

        out = {"K": K}

        if return_cov:
            # 用最后一次迭代的 H 近似协方差（注意维度随 DoF 变化）
            eps = 1e-6
            I = torch.eye(H.shape[-1], device=H.device).unsqueeze(0) * eps
            cov = torch.linalg.inv(H + I)
            out["cov"] = cov

        return out

