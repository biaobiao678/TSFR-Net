"""TSFR-Net: task-specific feature routing for RGB-D food nutrition estimation.

The public implementation uses paper-aligned module names. Legacy experiment
variant strings are intentionally retained so previously trained experiments
remain reproducible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from DFormer import DFormer_Base
from multiscale_refinement import MultiScaleRefinementPath
from feature_aggregation import CrossScaleFeatureAggregation


class AFGate(nn.Module):
    """Residual attention gate for DFormer RGB-D joint features."""
    def __init__(self, channels=(128, 256, 512, 1024), reduction=16):
        super().__init__()
        self.channel_gate = nn.ModuleList()
        self.spatial_gate = nn.ModuleList()
        for c in channels:
            hidden = max(c // reduction, 8)
            self.channel_gate.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, c, 1, bias=False),
                nn.Sigmoid()
            ))
            self.spatial_gate.append(nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid()
            ))
        # init as identity: F_out = F at the beginning of training
        self.alpha = nn.Parameter(torch.zeros(len(channels)))

    def forward(self, feats):
        outs = []
        for i, x in enumerate(feats):
            c_gate = self.channel_gate[i](x)
            avg_map = torch.mean(x, dim=1, keepdim=True)
            max_map, _ = torch.max(x, dim=1, keepdim=True)
            s_gate = self.spatial_gate[i](torch.cat([avg_map, max_map], dim=1))
            gate = c_gate * s_gate
            outs.append(x * (1.0 + self.alpha[i] * gate))
        return outs


class CJDE(nn.Module):
    """Complementary Joint Detail Enhancement for two DIE paths.
    The two inputs are both DFormer joint RGB-D features, not separated RGB/depth features.
    """
    def __init__(self, channel=256):
        super().__init__()
        self.a_from_b = nn.ModuleList([nn.Conv2d(channel, channel, 1) for _ in range(4)])
        self.b_from_a = nn.ModuleList([nn.Conv2d(channel, channel, 1) for _ in range(4)])
        self.alpha = nn.Parameter(torch.zeros(4))
        self.beta = nn.Parameter(torch.zeros(4))

    def forward(self, a_feats, b_feats):
        a_out, b_out = [], []
        for i, (a, b) in enumerate(zip(a_feats, b_feats)):
            gate_a = torch.sigmoid(self.a_from_b[i](b))
            gate_b = torch.sigmoid(self.b_from_a[i](a))
            a_out.append(a * (1.0 + self.alpha[i] * gate_a))
            b_out.append(b * (1.0 + self.beta[i] * gate_b))
        return tuple(a_out), tuple(b_out)


class RCMAFusion(nn.Module):
    """Residual Complementary Multiplicative Aggregation.
    More stable than pure multiplication.
    """
    def __init__(self, channel=256):
        super().__init__()
        self.norm_a = nn.ModuleList([nn.BatchNorm2d(channel) for _ in range(4)])
        self.norm_b = nn.ModuleList([nn.BatchNorm2d(channel) for _ in range(4)])
        self.beta = nn.Parameter(torch.zeros(4))

    def forward(self, a_feats, b_feats):
        fused = []
        for i, (a, b) in enumerate(zip(a_feats, b_feats)):
            mul = self.norm_a[i](a) * self.norm_b[i](b)
            fused.append(0.5 * (a + b) + self.beta[i] * mul)
        return tuple(fused)


class TopDownRefinement(nn.Module):
    """Top-down refinement after features have already been fused."""
    def __init__(self, channel=256):
        super().__init__()
        self.f1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.f2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.f3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.f4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.c1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.c2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.c3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.smooth1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.smooth2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.smooth3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        x1, x2, x3, x4 = feats
        x1 = self.relu(x1 + self.f1(x1))
        x2 = self.relu(x2 + self.f2(x2))
        x3 = self.relu(x3 + self.f3(x3))
        x4 = self.relu(x4 + self.f4(x4))

        t = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        t = self.relu(t + self.c3(t))
        x3 = x3 + t

        t = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        t = self.relu(t + self.c2(t))
        x2 = x2 + t

        t = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        t = self.relu(t + self.c1(t))
        x1 = x1 + t

        x1 = self.smooth1(x1)
        x2 = self.smooth2(x2)
        x3 = self.smooth3(x3)
        return x1, x2, x3, x4


class BidirectionalBinding(nn.Module):
    """Bidirectional binding between shallow detail feature x1 and deep semantic feature x4."""
    def __init__(self, channel=256):
        super().__init__()
        self.sem_to_det = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.det_to_sem = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.out1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.out4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x4):
        sem = self.sem_to_det(x4)
        sem = F.interpolate(sem, size=x1.shape[2:], mode='bilinear', align_corners=False)
        det = self.det_to_sem(x1)
        det = F.interpolate(det, size=x4.shape[2:], mode='bilinear', align_corners=False)

        x1_new = self.relu(x1 + self.gamma1 * self.out1(sem))
        x4_new = self.relu(x4 + self.gamma4 * self.out4(det))
        return x1_new, x4_new



class DetailToSemanticBranch(nn.Module):
    """One-way detail-to-semantic binding.
    Keep x1 unchanged to avoid hurting geometry/detail features.
    Only use shallow detail x1 to enhance deep semantic x4.
    """
    def __init__(self, channel=256):
        super().__init__()
        self.det_to_sem = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.out4 = nn.Conv2d(channel, channel, 3, 1, 1)
        # identity initialization: x4_new == x4 at the start of training
        self.gamma4 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x4):
        det = self.det_to_sem(x1)
        det = F.interpolate(det, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x4_new = self.relu(x4 + self.gamma4 * self.out4(det))
        return x1, x4_new


class DepthGeometryToken(nn.Module):
    """Use only the input depth map itself to produce a lightweight geometry token."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, depth):
        # depth: [B,1,H,W], already normalized by dataset transform
        b = depth.size(0)
        d = depth.view(b, -1)
        mean = d.mean(dim=1, keepdim=True)
        std = d.std(dim=1, keepdim=True)
        dmin = d.min(dim=1, keepdim=True)[0]
        dmax = d.max(dim=1, keepdim=True)[0]
        drange = dmax - dmin
        high_ratio = (d > mean).float().mean(dim=1, keepdim=True)
        stats = torch.cat([mean, std, dmin, dmax, drange, high_ratio], dim=1)
        return self.mlp(stats)


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StandardHeads(nn.Module):
    def __init__(self, in_dim=512, hidden=1024):
        super().__init__()
        self.cal_head = MLPHead(in_dim, hidden)
        self.mass_head = MLPHead(in_dim, hidden)
        self.fat_head = MLPHead(in_dim, hidden)
        self.carb_head = MLPHead(in_dim, hidden)
        self.pro_head = MLPHead(in_dim, hidden)

    def forward(self, feat):
        cal = self.cal_head(feat)
        mass = self.mass_head(feat)
        fat = self.fat_head(feat)
        carb = self.carb_head(feat)
        pro = self.pro_head(feat)
        return cal, mass, fat, carb, pro




class TaskSpecificHeads(nn.Module):
    """Five independent heads with task-specific input dimensions."""
    def __init__(self, in_dims, hidden=1024):
        super().__init__()
        self.cal_head = MLPHead(in_dims['cal'], hidden)
        self.mass_head = MLPHead(in_dims['mass'], hidden)
        self.fat_head = MLPHead(in_dims['fat'], hidden)
        self.carb_head = MLPHead(in_dims['carb'], hidden)
        self.pro_head = MLPHead(in_dims['pro'], hidden)

    def forward(self, feat_dict):
        cal = self.cal_head(feat_dict['cal'])
        mass = self.mass_head(feat_dict['mass'])
        fat = self.fat_head(feat_dict['fat'])
        carb = self.carb_head(feat_dict['carb'])
        pro = self.pro_head(feat_dict['pro'])
        return cal, mass, fat, carb, pro


class ExpertGatedRouteHead(nn.Module):
    """Task-specific adaptive fusion over several internal feature experts.

    Each expert feature is 512-dim: concat(GAP(x1), GAP(x4)).
    For every nutrient task, a small gate learns how much to use from
    base / Det2Sem / RCMA / CJDE features. This uses no external information.
    """
    def __init__(self, expert_names, feat_dim=512, hidden=1024, gate_hidden=256):
        super().__init__()
        self.expert_names = list(expert_names)
        self.tasks = ['cal', 'mass', 'fat', 'carb', 'pro']
        gate_in = feat_dim * len(self.expert_names)
        self.gates = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(gate_in, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, len(self.expert_names))
            ) for t in self.tasks
        })
        self.heads = nn.ModuleDict({t: MLPHead(feat_dim, hidden) for t in self.tasks})

    def _fuse(self, task, expert_feats):
        # expert_feats: dict name -> [B, 512]
        cat = torch.cat([expert_feats[name] for name in self.expert_names], dim=1)
        w = torch.softmax(self.gates[task](cat), dim=1)  # [B, E]
        stacked = torch.stack([expert_feats[name] for name in self.expert_names], dim=1)  # [B, E, 512]
        fused = (w.unsqueeze(-1) * stacked).sum(dim=1)
        return fused

    def forward(self, expert_feats):
        cal = self.heads['cal'](self._fuse('cal', expert_feats))
        mass = self.heads['mass'](self._fuse('mass', expert_feats))
        fat = self.heads['fat'](self._fuse('fat', expert_feats))
        carb = self.heads['carb'](self._fuse('carb', expert_feats))
        pro = self.heads['pro'](self._fuse('pro', expert_feats))
        return cal, mass, fat, carb, pro

class SoftTAHead(nn.Module):
    """Task-aware soft gated head. Each task adaptively uses x1 and x4."""
    def __init__(self, channel=256, geo_dim=0, hidden=1024):
        super().__init__()
        self.geo_dim = geo_dim
        gate_in = channel * 2 + geo_dim
        head_in = channel * 2 + geo_dim
        self.tasks = ['cal', 'mass', 'fat', 'carb', 'pro']
        self.gates = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(gate_in, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)
            ) for t in self.tasks
        })
        self.heads = nn.ModuleDict({t: MLPHead(head_in, hidden) for t in self.tasks})

    def _task_feat(self, task, z1, z4, geo=None):
        gate_base = torch.cat([z1, z4], dim=1) if geo is None else torch.cat([z1, z4, geo], dim=1)
        w = torch.softmax(self.gates[task](gate_base), dim=1)
        feat = torch.cat([w[:, 0:1] * z1, w[:, 1:2] * z4], dim=1)
        if geo is not None:
            feat = torch.cat([feat, geo], dim=1)
        return feat

    def forward(self, z1, z4, geo=None):
        cal = self.heads['cal'](self._task_feat('cal', z1, z4, geo))
        mass = self.heads['mass'](self._task_feat('mass', z1, z4, geo))
        fat = self.heads['fat'](self._task_feat('fat', z1, z4, geo))
        carb = self.heads['carb'](self._task_feat('carb', z1, z4, geo))
        pro = self.heads['pro'](self._task_feat('pro', z1, z4, geo))
        return cal, mass, fat, carb, pro


class MassDensityTAHead(nn.Module):
    """Mass-density decoupled prediction head.
    This is more experimental than SoftTAHead. Use after v3 is verified.
    """
    def __init__(self, channel=256, geo_dim=0, hidden=1024):
        super().__init__()
        self.geo_dim = geo_dim
        gate_in = channel * 2 + geo_dim
        head_in = channel * 2 + geo_dim
        self.tasks = ['mass', 'cal_d', 'fat_d', 'carb_d', 'pro_d']
        self.gates = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(gate_in, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)
            ) for t in self.tasks
        })
        self.heads = nn.ModuleDict({t: MLPHead(head_in, hidden) for t in self.tasks})
        self.softplus = nn.Softplus()

    def _task_feat(self, task, z1, z4, geo=None):
        gate_base = torch.cat([z1, z4], dim=1) if geo is None else torch.cat([z1, z4, geo], dim=1)
        w = torch.softmax(self.gates[task](gate_base), dim=1)
        feat = torch.cat([w[:, 0:1] * z1, w[:, 1:2] * z4], dim=1)
        if geo is not None:
            feat = torch.cat([feat, geo], dim=1)
        return feat

    def forward(self, z1, z4, geo=None):
        mass = self.softplus(self.heads['mass'](self._task_feat('mass', z1, z4, geo))) + 1e-6
        cal_d = self.softplus(self.heads['cal_d'](self._task_feat('cal_d', z1, z4, geo)))
        fat_d = self.softplus(self.heads['fat_d'](self._task_feat('fat_d', z1, z4, geo)))
        carb_d = self.softplus(self.heads['carb_d'](self._task_feat('carb_d', z1, z4, geo)))
        pro_d = self.softplus(self.heads['pro_d'](self._task_feat('pro_d', z1, z4, geo)))
        cal = mass * cal_d
        fat = mass * fat_d
        carb = mass * carb_d
        pro = mass * pro_d
        return cal, mass, fat, carb, pro



class MassResidualCorrection(nn.Module):
    """Zero-initialized all-scale residual branch for Mass.
    Input is concat(GAP(x1), GAP(x2), GAP(x3), GAP(x4)) = 1024 dim.
    The last layer is initialized to zero, so the branch initially outputs 0.
    """
    def __init__(self, in_dim=1024, hidden=512, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class CarbResidualCorrection(nn.Module):
    """Zero-initialized wide residual adapter for Carb.
    Input is the v31 base feature concat(GAP(x1_base), GAP(x4_base)) = 512 dim.
    """
    def __init__(self, in_dim=512, hidden=384, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class TSFRNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.exp_variant = getattr(args, 'exp_variant', 'v4')

        self.backbone = DFormer_Base()

        self.adapt1 = nn.Conv2d(64, 128, 1)
        self.adapt2 = nn.Conv2d(128, 256, 1)
        self.adapt3 = nn.Conv2d(256, 512, 1)
        self.adapt4 = nn.Conv2d(512, 1024, 1)

        # Old ablation variants are kept for compatibility.
        self.use_af = self.exp_variant in [
            'v3_af_bgsie_tahead',
            'v4_af_cjde_rcma_bgsie_tahead',
            'v5_af_bgsie_md',
            'v6_af_bgsie_geo_md'
        ]
        self.use_bgsie = self.exp_variant in [
            'v1_bgsie',
            'v2_bgsie_tahead',
            'v3_af_bgsie_tahead',
            'v4_af_cjde_rcma_bgsie_tahead',
            'v5_af_bgsie_md',
            'v6_af_bgsie_geo_md'
        ]
        self.use_tahead = self.exp_variant in [
            'v2_bgsie_tahead',
            'v3_af_bgsie_tahead',
            'v4_af_cjde_rcma_bgsie_tahead'
        ]
        self.use_md = self.exp_variant in ['v5_af_bgsie_md', 'v6_af_bgsie_geo_md']
        self.use_geo = self.exp_variant == 'v6_af_bgsie_geo_md'

        # Previous next-round variants.
        self.use_cjde = self.exp_variant in [
            'v4_af_cjde_rcma_bgsie_tahead',
            'v7_cjde_rcma_std',
            'v9_cjde_sie_std',
            'v11_cjde_rcma_det2sem_std',
            'v18_cjde_massprotect',
            'v19_cjde_calmassprotect',
            'v20_cjde_x4skip',
            'v23_route_det_cjde_best',
            'v25_route_det_cjde_rcma_best',
            'v26_route_det_cjde_rcma_safe',
            'v27_route_det_cjde_altfat',
            'v30_v16_carb_cjde_protein_rcma',
            'v35_v16_carb_cjde_fatmix_protein_rcma'
        ,
            'v40_v31_mass_cjde',
            'v42_v31_carb_det_mass_cjde',
            'v48_adaptive_base_det_cjde_rcma']
        self.use_rcma = self.exp_variant in [
            'v4_af_cjde_rcma_bgsie_tahead',
            'v7_cjde_rcma_std',
            'v8_rcma_std',
            'v11_cjde_rcma_det2sem_std',
            'v22_rcma_taskprotect',
            'v24_route_det_rcma_best',
            'v25_route_det_cjde_rcma_best',
            'v26_route_det_cjde_rcma_safe',
            'v27_route_det_cjde_altfat',
            'v29_v16_protein_rcma',
            'v30_v16_carb_cjde_protein_rcma',
            'v31_v16_cal_det_protein_rcma',
            'v32_v16_carb_det_protein_rcma',
            'v34_v16_protein_rcma_fatmix',
            'v35_v16_carb_cjde_fatmix_protein_rcma',
            'v37_protein_rcma_only'
        ,
            'v39_v31_mass_rcma',
            'v41_v31_carb_det_mass_rcma',
            'v43_v31_protein_mix',
            'v45_v31_mass_mix',
            'v46_v31_mass_carb_pro_mix',
            'v47_adaptive_base_det_rcma',
            'v48_adaptive_base_det_cjde_rcma',
            'v38_v31_carb_det',
            'v39_v31_mass_rcma',
            'v40_v31_mass_cjde',
            'v41_v31_carb_det_mass_rcma',
            'v42_v31_carb_det_mass_cjde',
            'v43_v31_protein_mix',
            'v44_v31_carb_mix',
            'v45_v31_mass_mix',
            'v46_v31_mass_carb_pro_mix',
            'v47_adaptive_base_det_rcma',
            'v48_adaptive_base_det_cjde_rcma',
            'v49_adaptive_base_det',
            'v57_v31_mass_allscale_residual',
            'v59_v31_carb_wide_adapter',
            'v60_v31_mass_allscale_residual_carb_wide']
        self.use_det2sem = self.exp_variant in [
            'v10_det2sem_std',
            'v11_cjde_rcma_det2sem_std',
            'v12_det2sem_massprotect',
            'v13_det2sem_x4skip',
            'v14_det2sem_mass_carbprotect',
            'v15_det2sem_mass_x1only',
            'v16_det2sem_fatonly',
            'v17_det2sem_cal_fat_protect',
            'v23_route_det_cjde_best',
            'v24_route_det_rcma_best',
            'v25_route_det_cjde_rcma_best',
            'v26_route_det_cjde_rcma_safe',
            'v27_route_det_cjde_altfat',
            'v28_v16_detach_fat',
            'v29_v16_protein_rcma',
            'v30_v16_carb_cjde_protein_rcma',
            'v31_v16_cal_det_protein_rcma',
            'v32_v16_carb_det_protein_rcma',
            'v33_v16_fatmix_base_det',
            'v34_v16_protein_rcma_fatmix',
            'v35_v16_carb_cjde_fatmix_protein_rcma',
            'v36_v16_carbpro_det2sem'
        ,
            'v38_v31_carb_det',
            'v39_v31_mass_rcma',
            'v40_v31_mass_cjde',
            'v41_v31_carb_det_mass_rcma',
            'v42_v31_carb_det_mass_cjde',
            'v43_v31_protein_mix',
            'v44_v31_carb_mix',
            'v45_v31_mass_mix',
            'v46_v31_mass_carb_pro_mix',
            'v47_adaptive_base_det_rcma',
            'v48_adaptive_base_det_cjde_rcma',
            'v49_adaptive_base_det',
            'v57_v31_mass_allscale_residual',
            'v59_v31_carb_wide_adapter',
            'v60_v31_mass_allscale_residual_carb_wide']

        self.route_variants = {
            'v12_det2sem_massprotect',
            'v13_det2sem_x4skip',
            'v14_det2sem_mass_carbprotect',
            'v15_det2sem_mass_x1only',
            'v16_det2sem_fatonly',
            'v17_det2sem_cal_fat_protect',
            'v18_cjde_massprotect',
            'v19_cjde_calmassprotect',
            'v20_cjde_x4skip',
            'v22_rcma_taskprotect',
            'v23_route_det_cjde_best',
            'v24_route_det_rcma_best',
            'v25_route_det_cjde_rcma_best',
            'v26_route_det_cjde_rcma_safe',
            'v27_route_det_cjde_altfat',
            'v28_v16_detach_fat',
            'v29_v16_protein_rcma',
            'v30_v16_carb_cjde_protein_rcma',
            'v31_v16_cal_det_protein_rcma',
            'v32_v16_carb_det_protein_rcma',
            'v33_v16_fatmix_base_det',
            'v34_v16_protein_rcma_fatmix',
            'v35_v16_carb_cjde_fatmix_protein_rcma',
            'v36_v16_carbpro_det2sem',
            'v37_protein_rcma_only',
            'v57_v31_mass_allscale_residual',
            'v59_v31_carb_wide_adapter',
            'v60_v31_mass_allscale_residual_carb_wide'
        }
        self.use_route_head = self.exp_variant in self.route_variants

        if self.use_af:
            self.af_gate = AFGate(channels=(128, 256, 512, 1024))

        self.refine_path_a = MultiScaleRefinementPath(dim=256)
        self.refine_path_b = MultiScaleRefinementPath(dim=256)

        if self.use_cjde:
            self.cjde = CJDE(channel=256)
        if self.use_rcma:
            self.rcma = RCMAFusion(channel=256)
            self.rcma_refinement = TopDownRefinement(channel=256)

        # Base cross-scale aggregation is still needed by v4/v9/v10/v12-v20 and route variants.
        if (not self.use_rcma) or self.use_route_head or self.exp_variant in ['v4_af_cjde_rcma_bgsie_tahead']:
            self.base_aggregation = CrossScaleFeatureAggregation(channel=256)

        if self.use_bgsie:
            self.bidirectional_binding = BidirectionalBinding(channel=256)
        if self.use_det2sem:
            self.detail_to_semantic = DetailToSemanticBranch(channel=256)

        self.pool = nn.AdaptiveAvgPool2d(1)

        geo_dim = 0
        if self.use_geo:
            geo_dim = getattr(args, 'geo_dim', 64)
            self.geo_token = DepthGeometryToken(out_dim=geo_dim)

        if self.use_route_head:
            self.head = self._build_route_head()
        elif self.use_md:
            self.head = MassDensityTAHead(channel=256, geo_dim=geo_dim, hidden=1024)
        elif self.use_tahead:
            self.head = SoftTAHead(channel=256, geo_dim=geo_dim, hidden=1024)
        else:
            self.head = StandardHeads(in_dim=512 + geo_dim, hidden=1024)

        # Epoch-146 safe residual experiments. These branches are zero-initialized,
        # so the initial forward result is exactly the same as v31.
        self.use_mass_allscale_residual = self.exp_variant in [
            'v57_v31_mass_allscale_residual',
            'v60_v31_mass_allscale_residual_carb_wide'
        ]
        self.use_carb_wide_residual = self.exp_variant in [
            'v59_v31_carb_wide_adapter',
            'v60_v31_mass_allscale_residual_carb_wide'
        ]
        if self.use_mass_allscale_residual:
            self.mass_residual_correction = MassResidualCorrection(in_dim=1024, hidden=512, dropout=0.10)
        if self.use_carb_wide_residual:
            self.carb_residual_correction = CarbResidualCorrection(in_dim=512, hidden=384, dropout=0.10)

    def _build_route_head(self):
        d512 = 512
        d768 = 768
        d256 = 256
        v = self.exp_variant

        if v in ['v13_det2sem_x4skip', 'v20_cjde_x4skip']:
            return StandardHeads(in_dim=d768, hidden=1024)

        if v == 'v15_det2sem_mass_x1only':
            return TaskSpecificHeads({'cal': d512, 'mass': d256, 'fat': d512, 'carb': d512, 'pro': d512}, hidden=1024)

        # New v16-centered variants with fat feature concatenation.
        if v in ['v33_v16_fatmix_base_det', 'v34_v16_protein_rcma_fatmix', 'v35_v16_carb_cjde_fatmix_protein_rcma']:
            return TaskSpecificHeads({'cal': d512, 'mass': d512, 'fat': 1024, 'carb': d512, 'pro': d512}, hidden=1024)

        # v31 refinement variants with feature concatenation for selected tasks.
        if v == 'v43_v31_protein_mix':
            return TaskSpecificHeads({'cal': d512, 'mass': d512, 'fat': d512, 'carb': d512, 'pro': 1024}, hidden=1024)

        if v == 'v44_v31_carb_mix':
            return TaskSpecificHeads({'cal': d512, 'mass': d512, 'fat': d512, 'carb': 1024, 'pro': d512}, hidden=1024)

        if v == 'v45_v31_mass_mix':
            return TaskSpecificHeads({'cal': d512, 'mass': 1024, 'fat': d512, 'carb': d512, 'pro': d512}, hidden=1024)

        if v == 'v46_v31_mass_carb_pro_mix':
            return TaskSpecificHeads({'cal': d512, 'mass': 1024, 'fat': d512, 'carb': 1024, 'pro': 1024}, hidden=1024)

        if v == 'v47_adaptive_base_det_rcma':
            return ExpertGatedRouteHead(['base', 'det', 'rcma'], feat_dim=512, hidden=1024)

        if v == 'v48_adaptive_base_det_cjde_rcma':
            return ExpertGatedRouteHead(['base', 'det', 'cjde', 'rcma'], feat_dim=512, hidden=1024)

        if v == 'v49_adaptive_base_det':
            return ExpertGatedRouteHead(['base', 'det'], feat_dim=512, hidden=1024)

        # All other route variants use 512-dim features for all tasks.
        return TaskSpecificHeads({'cal': d512, 'mass': d512, 'fat': d512, 'carb': d512, 'pro': d512}, hidden=1024)

    @staticmethod
    def remap_legacy_state_dict(state_dict):
        """Map checkpoints saved with the earlier ADFE-style attribute names.

        This only changes parameter keys; tensor values are untouched.
        """
        replacements = {
            'MSF1.': 'refine_path_a.',
            'MSF2.': 'refine_path_b.',
            'rgbdf.': 'base_aggregation.',
            'single_sie.': 'rcma_refinement.',
            'bg_sie.': 'bidirectional_binding.',
            'det2sem.': 'detail_to_semantic.',
            'mass_allscale_residual.': 'mass_residual_correction.',
            'carb_wide_residual.': 'carb_residual_correction.',
        }
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in replacements.items():
                if new_key.startswith(old_prefix):
                    new_key = new_prefix + new_key[len(old_prefix):]
                    break
            remapped[new_key] = value
        return remapped

    def init_weights(self, pretrained):
        self.backbone.init_weights(pretrained)

    def _adapt_feats(self, feats):
        feats = [
            self.adapt1(feats[0]),
            self.adapt2(feats[1]),
            self.adapt3(feats[2]),
            self.adapt4(feats[3])
        ]
        if self.use_af:
            feats = self.af_gate(feats)
        return feats

    def _pool_pair(self, x1, x4):
        z1 = self.pool(x1).flatten(1)
        z4 = self.pool(x4).flatten(1)
        return z1, z4, torch.cat([z1, z4], dim=1)

    def _standard_features(self, path_a, path_b):
        x1, x2, x3, x4 = self.base_aggregation(path_a, path_b)
        return x1, x4

    def _det_features(self, path_a, path_b):
        x1_base, x4_base = self._standard_features(path_a, path_b)
        x1_det, x4_det = self.detail_to_semantic(x1_base, x4_base)
        return x1_base, x4_base, x1_det, x4_det

    def _cjde_features(self, path_a, path_b):
        cjde_a, cjde_b = self.cjde(path_a, path_b)
        x1_cjde, x4_cjde = self._standard_features(cjde_a, cjde_b)
        return x1_cjde, x4_cjde

    def _rcma_features(self, path_a, path_b):
        fused = self.rcma(path_a, path_b)
        x1, x2, x3, x4 = self.rcma_refinement(fused)
        return x1, x4

    def _extract_x1_x4(self, feats):
        path_a = self.refine_path_a(feats)
        path_b = self.refine_path_b(feats)

        if self.use_cjde:
            path_a, path_b = self.cjde(path_a, path_b)

        if self.use_rcma:
            fused = self.rcma(path_a, path_b)
            x1, x2, x3, x4 = self.rcma_refinement(fused)
        else:
            x1, x2, x3, x4 = self.base_aggregation(path_a, path_b)

        if self.use_bgsie:
            x1, x4 = self.bidirectional_binding(x1, x4)
        if self.use_det2sem:
            x1, x4 = self.detail_to_semantic(x1, x4)
        return x1, x4

    def _forward_route(self, feats):
        path_a = self.refine_path_a(feats)
        path_b = self.refine_path_b(feats)
        v = self.exp_variant

        # Base v4 features. Keep x2/x3 for the v57/v60 Mass all-scale residual branch.
        x1_base, x2_base, x3_base, x4_base = self.base_aggregation(path_a, path_b)
        z1_base, z4_base, feat_base = self._pool_pair(x1_base, x4_base)

        feat_det = None
        feat_cjde = None
        feat_rcma = None
        z1_det = None
        z4_det = None
        z1_cjde = None
        z4_cjde = None
        z1_rcma = None
        z4_rcma = None

        if self.use_det2sem:
            # v28 blocks the Det2Sem/fat branch gradient from changing the shared backbone.
            # This tests whether v16's mass degradation is caused by Det2Sem gradients.
            if v == 'v28_v16_detach_fat':
                x1_det, x4_det = self.detail_to_semantic(x1_base.detach(), x4_base.detach())
            else:
                x1_det, x4_det = self.detail_to_semantic(x1_base, x4_base)
            z1_det, z4_det, feat_det = self._pool_pair(x1_det, x4_det)

        if self.use_cjde:
            x1_cjde, x4_cjde = self._cjde_features(path_a, path_b)
            z1_cjde, z4_cjde, feat_cjde = self._pool_pair(x1_cjde, x4_cjde)

        if self.use_rcma:
            x1_rcma, x4_rcma = self._rcma_features(path_a, path_b)
            z1_rcma, z4_rcma, feat_rcma = self._pool_pair(x1_rcma, x4_rcma)

        if v == 'v12_det2sem_massprotect':
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_det, 'pro': feat_det})

        if v == 'v13_det2sem_x4skip':
            feat_skip = torch.cat([z1_base, z4_base, z4_det], dim=1)
            return self.head(feat_skip)

        if v == 'v14_det2sem_mass_carbprotect':
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_det})

        if v == 'v15_det2sem_mass_x1only':
            return self.head({'cal': feat_det, 'mass': z1_base, 'fat': feat_det, 'carb': feat_det, 'pro': feat_det})

        if v == 'v16_det2sem_fatonly':
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_base})

        if v == 'v17_det2sem_cal_fat_protect':
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_base})

        if v == 'v18_cjde_massprotect':
            return self.head({'cal': feat_cjde, 'mass': feat_base, 'fat': feat_cjde, 'carb': feat_cjde, 'pro': feat_cjde})

        if v == 'v19_cjde_calmassprotect':
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_cjde, 'carb': feat_cjde, 'pro': feat_cjde})

        if v == 'v20_cjde_x4skip':
            feat_skip = torch.cat([z1_base, z4_base, z4_cjde], dim=1)
            return self.head(feat_skip)

        if v == 'v22_rcma_taskprotect':
            return self.head({'cal': feat_rcma, 'mass': feat_base, 'fat': feat_base, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v23_route_det_cjde_best':
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_cjde, 'pro': feat_cjde})

        if v == 'v24_route_det_rcma_best':
            return self.head({'cal': feat_rcma, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v25_route_det_cjde_rcma_best':
            return self.head({'cal': feat_rcma, 'mass': feat_base, 'fat': feat_det, 'carb': feat_cjde, 'pro': feat_rcma})

        if v == 'v26_route_det_cjde_rcma_safe':
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_cjde, 'pro': feat_rcma})

        if v == 'v27_route_det_cjde_altfat':
            return self.head({'cal': feat_rcma, 'mass': feat_base, 'fat': feat_cjde, 'carb': feat_cjde, 'pro': feat_rcma})

        # --------------------------
        # v16-centered refinement variants.
        # Current best: v16_det2sem_fatonly. These variants keep its useful part
        # and only change mass/protein/fat/carb routing in a controlled way.
        # --------------------------
        if v == 'v28_v16_detach_fat':
            # Same visible route as v16, but Det2Sem branch is detached from shared features.
            # Goal: protect mass by preventing fat/Det2Sem gradients from perturbing v4 features.
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_base})

        if v == 'v29_v16_protein_rcma':
            # v16 + protein uses RCMA, because v24 showed RCMA is strong for protein.
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v30_v16_carb_cjde_protein_rcma':
            # v16 + carb uses CJDE and protein uses RCMA.
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_det, 'carb': feat_cjde, 'pro': feat_rcma})

        if v == 'v31_v16_cal_det_protein_rcma':
            # Det2Sem for cal/fat, RCMA for protein, base for mass/carb.
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v32_v16_carb_det_protein_rcma':
            # Det2Sem for fat/carb, RCMA for protein, base for cal/mass.
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_det, 'carb': feat_det, 'pro': feat_rcma})

        if v == 'v33_v16_fatmix_base_det':
            # Fat head receives both base and Det2Sem features. Other tasks follow v16's base route.
            feat_fat = torch.cat([feat_base, feat_det], dim=1)
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_fat, 'carb': feat_base, 'pro': feat_base})

        if v == 'v34_v16_protein_rcma_fatmix':
            # v33 + protein uses RCMA.
            feat_fat = torch.cat([feat_base, feat_det], dim=1)
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_fat, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v35_v16_carb_cjde_fatmix_protein_rcma':
            # Fat gets base+Det2Sem, carb gets CJDE, protein gets RCMA.
            feat_fat = torch.cat([feat_base, feat_det], dim=1)
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_fat, 'carb': feat_cjde, 'pro': feat_rcma})

        if v == 'v36_v16_carbpro_det2sem':
            # Det2Sem is used for fat/carb/protein, while cal/mass are protected by base features.
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_det, 'carb': feat_det, 'pro': feat_det})

        if v == 'v37_protein_rcma_only':
            # Isolate RCMA's protein contribution without Det2Sem.
            return self.head({'cal': feat_base, 'mass': feat_base, 'fat': feat_base, 'carb': feat_base, 'pro': feat_rcma})

        # --------------------------
        # v31-centered refinement variants.
        # Current best: v31_v16_cal_det_protein_rcma.
        # v31 route: cal=Det2Sem, mass=base, fat=Det2Sem, carb=base, protein=RCMA.
        # These variants try to repair carb/mass/protein without losing the fat gain.
        # --------------------------
        if v == 'v38_v31_carb_det':
            # v31 + carb also uses Det2Sem. Tests whether Det2Sem can recover carb while keeping v31 fat.
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_det, 'pro': feat_rcma})

        if v == 'v39_v31_mass_rcma':
            # v31 + mass uses RCMA. Tests whether RCMA can repair mass while preserving v31 routing.
            return self.head({'cal': feat_det, 'mass': feat_rcma, 'fat': feat_det, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v40_v31_mass_cjde':
            # v31 + mass uses CJDE. Tests whether CJDE is a better mass path than RCMA.
            return self.head({'cal': feat_det, 'mass': feat_cjde, 'fat': feat_det, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v41_v31_carb_det_mass_rcma':
            # Combine carb Det2Sem and mass RCMA.
            return self.head({'cal': feat_det, 'mass': feat_rcma, 'fat': feat_det, 'carb': feat_det, 'pro': feat_rcma})

        if v == 'v42_v31_carb_det_mass_cjde':
            # Combine carb Det2Sem and mass CJDE.
            return self.head({'cal': feat_det, 'mass': feat_cjde, 'fat': feat_det, 'carb': feat_det, 'pro': feat_rcma})

        if v == 'v43_v31_protein_mix':
            # Protein receives both base and RCMA features. Other tasks stay as v31.
            feat_pro = torch.cat([feat_base, feat_rcma], dim=1)
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_base, 'pro': feat_pro})

        if v == 'v44_v31_carb_mix':
            # Carb receives both base and Det2Sem features. Other tasks stay as v31.
            feat_carb = torch.cat([feat_base, feat_det], dim=1)
            return self.head({'cal': feat_det, 'mass': feat_base, 'fat': feat_det, 'carb': feat_carb, 'pro': feat_rcma})

        if v == 'v45_v31_mass_mix':
            # Mass receives both base and RCMA features. Other tasks stay as v31.
            feat_mass = torch.cat([feat_base, feat_rcma], dim=1)
            return self.head({'cal': feat_det, 'mass': feat_mass, 'fat': feat_det, 'carb': feat_base, 'pro': feat_rcma})

        if v == 'v46_v31_mass_carb_pro_mix':
            # A safer mixed version: mass=base+RCMA, carb=base+Det2Sem, protein=base+RCMA.
            feat_mass = torch.cat([feat_base, feat_rcma], dim=1)
            feat_carb = torch.cat([feat_base, feat_det], dim=1)
            feat_pro = torch.cat([feat_base, feat_rcma], dim=1)
            return self.head({'cal': feat_det, 'mass': feat_mass, 'fat': feat_det, 'carb': feat_carb, 'pro': feat_pro})

        if v == 'v47_adaptive_base_det_rcma':
            # Task-specific learned fusion over base / Det2Sem / RCMA.
            return self.head({'base': feat_base, 'det': feat_det, 'rcma': feat_rcma})

        if v == 'v48_adaptive_base_det_cjde_rcma':
            # Task-specific learned fusion over base / Det2Sem / CJDE / RCMA.
            return self.head({'base': feat_base, 'det': feat_det, 'cjde': feat_cjde, 'rcma': feat_rcma})

        if v == 'v49_adaptive_base_det':
            # Conservative adaptive fusion over base / Det2Sem only.
            return self.head({'base': feat_base, 'det': feat_det})

        if v in [
            'v57_v31_mass_allscale_residual',
            'v59_v31_carb_wide_adapter',
            'v60_v31_mass_allscale_residual_carb_wide'
        ]:
            # Start from the exact v31 route:
            # cal=Det2Sem, mass=base, fat=Det2Sem, carb=base, protein=RCMA.
            cal, mass, fat, carb, pro = self.head({
                'cal': feat_det,
                'mass': feat_base,
                'fat': feat_det,
                'carb': feat_base,
                'pro': feat_rcma
            })

            if self.use_mass_allscale_residual:
                feat_mass_all = torch.cat([
                    self.pool(x1_base),
                    self.pool(x2_base),
                    self.pool(x3_base),
                    self.pool(x4_base)
                ], dim=1).flatten(1)
                mass = mass + self.mass_residual_correction(feat_mass_all)

            if self.use_carb_wide_residual:
                carb = carb + self.carb_residual_correction(feat_base)

            return cal, mass, fat, carb, pro

        raise ValueError(f'Unknown route variant: {v}')

    def forward(self, rgb, depth):
        feats, _ = self.backbone(rgb, depth)
        feats = self._adapt_feats(feats)

        if self.use_route_head:
            return self._forward_route(feats)

        x1, x4 = self._extract_x1_x4(feats)
        z1 = self.pool(x1).flatten(1)
        z4 = self.pool(x4).flatten(1)
        geo = self.geo_token(depth) if self.use_geo else None

        if self.use_tahead or self.use_md:
            return self.head(z1, z4, geo)
        else:
            feat = torch.cat([z1, z4], dim=1)
            if geo is not None:
                feat = torch.cat([feat, geo], dim=1)
            return self.head(feat)


# Backward-compatible class alias for old training scripts/checkpoints.
TSFR = TSFRNet
