import time
import csv
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.optim.swa_utils import AveragedModel

import os
import torchvision.transforms as transforms
from dataloader import get_DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

#导入DFormer
from tsfr_net import TSFRNet

# 显存优化：开启TF32+benchmark（32G显存满负载核心）
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TSFRTrainer(object):
    def __init__(self, args):
        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.b
        self.gpu_mode = not args.cpu
        self.amp = args.amp
        self.seed = args.seed
        self.loss_path = args.loss_path
        self.dump_path = args.dump
        self.test_path = args.t_result
        self.load_dump = (args.load != '')
        self.load_path = Path(args.load)
        self.start_epoch = 1
        self.min_avg = 0.5
        self.best_epoch = 0

        self.accum_steps = 1
        self.resume = (args.resume != '')
        self.resume_path = Path(os.path.abspath(args.resume)) if self.resume else None

        # ============================================================
        # epoch146 安全微调模式：
        # 使用 --load 且非 --test / 非 --resume 时启用。
        # 只加载模型权重，不加载 optimizer，不延续旧 epoch。
        # 新增模块允许 strict=False，适合 v57/v59/v60 的 zero-init 残差分支。
        # 学习率使用长周期 cosine：T_max=args.epoch，60轮内持续下降，不再30轮回升。
        # ============================================================
        self.safe_load_finetune = self.load_dump and (not args.test) and (not self.resume)
        self.loaded_baseline_epoch = 0

        # ============================================================
        # 速度策略：
        # 1) 60轮以内默认不 torch.compile，避免第一轮编译卡很久。
        # 2) 如果训练超过60轮，第61轮开始自动编译训练模型，后续长训练更快。
        # 3) 测试阶段默认不走 compile，避免 eval 图重新编译导致测试变慢。
        # ============================================================
        self.compile_after_epoch = 60
        self.enable_delayed_compile = True
        self.compiled_train_model = False
        self.compile_failed = False

        #self.fast_test_amp = True
        self.fast_test_amp = False
        self.fast_test_use_compiled_model = False

        # ============================================================
        # 冻结式 adapter 微调：
        # 针对 v57/v59/v60，冻结 v31 主体，只训练新增 residual adapter。
        # 目的：保留 epoch146 的任务平衡，只让弱项输出做小幅残差修正。
        # ============================================================
        self.freeze_adapter_variants = {
            'v57_v31_mass_allscale_residual',
            'v59_v31_carb_wide_adapter',
            'v60_v31_mass_allscale_residual_carb_wide'
        }
        self.freeze_adapter_mode = False

        self.history = {
            "epoch": [], "train_loss": [], "test_mean_pmae": [],
            "cal_pmae": [], "mass_pmae": [], "fat_pmae": [],
            "carb_pmae": [], "protein_pmae": [], "lr": []
        }

        self.set_seed()
        self.train_data_loader, self.test_data_loader = get_DataLoader(args)

        self.model = TSFRNet(args=self.args)
        self.C_net = self.model

        if hasattr(args, 'pre') and args.pre != '':
            self.model.init_weights(args.pre)

        if args.test:
            for param in self.C_net.parameters():
                param.requires_grad = False

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

        # ============================================================
        # 学习率策略：
        # 1) 正常从头训练：保留原来的 T_max=30 策略。
        # 2) --resume 续训：长周期 cosine，T_max=args.epoch，并对齐已有轮数。
        # 3) --load 安全微调：长周期 cosine，T_max=args.epoch，从第1轮单调下降到 eta_min。
        # ============================================================
        if self.resume:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epoch,
                eta_min=1e-6
            )
            self.scheduler_mode = f"resume_long_cosine_Tmax_{args.epoch}"
        elif self.safe_load_finetune:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epoch,
                eta_min=1e-6
            )
            self.scheduler_mode = f"load_safe_long_cosine_Tmax_{args.epoch}"
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=30,
                eta_min=1e-6
            )
            self.scheduler_mode = "initial_original_cosine_Tmax_30"

        self.L1Loss = nn.L1Loss(reduction='sum')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)

        if self.gpu_mode:
            self.model.to(self.device)
            self.C_net.to(self.device)
            self.L1Loss.to(self.device)

        self.ema_model = AveragedModel(self.model)
        self.ema_update_freq = 10

        if self.gpu_mode and self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        if self.safe_load_finetune:
            self._load_for_safe_finetune()

        if self.resume:
            self._resume_with_long_cosine()

        if not args.test:
            if self.resume:
                self.dump_path = os.path.dirname(self.resume_path)
            else:
                variant = getattr(args, 'exp_variant', 'v4')
                t = time.strftime('%y%m%d-%H%M%S')
                self.dump_path = os.path.join(args.dump, f'{variant}_{t}')
            os.makedirs(self.dump_path, exist_ok=True)
            self.csv_path = os.path.join(self.dump_path, 'training_history.csv')
            self._load_existing_history_for_resume()

            # 安全微调时，先把加载后的初始模型保存为 best.pkl。
            # 这样如果60轮内没有超过 baseline，也不会出现没有 best.pkl 的情况。
            if self.safe_load_finetune:
                self.best_epoch = 0
                self.save(0)
                print(
                    f"✅ 已保存安全微调初始点 best.pkl | "
                    f"baseline_epoch={self.loaded_baseline_epoch} | baseline_min_avg={self.min_avg:.6f}"
                )

        self.tb_writer = None
        if not args.test:
            self.tb_log_dir = os.path.join(self.dump_path, "tb_logs")
            self.tb_writer = SummaryWriter(self.tb_log_dir)
            print(f"✅ TensorBoard 全指标已开启 | 保存路径：{self.tb_log_dir}")
            print(f"✅ LR策略：{self.scheduler_mode} | start_epoch={self.start_epoch} | current_lr={self.optimizer.param_groups[0]['lr']:.8f}")
            print(f"✅ 编译策略：前{self.compile_after_epoch}轮不编译，第{self.compile_after_epoch + 1}轮开始训练编译加速")
            print(f"✅ 测试策略：inference_mode + AMP={self.fast_test_amp} | use_compile={self.fast_test_use_compiled_model}")

    def _remove_module_prefix(self, state_dict):
        return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    def _infer_baseline_min_avg(self, checkpoint, checkpoint_path):
        baseline = checkpoint.get('min_avg', self.min_avg)
        finish_epoch = checkpoint.get('finish_epoch', 0)
        self.loaded_baseline_epoch = finish_epoch

        csv_path = Path(checkpoint_path).parent / 'training_history.csv'
        if not csv_path.exists():
            return baseline

        try:
            df = pd.read_csv(csv_path)
            if 'test_mean_pmae' not in df.columns:
                return baseline

            if finish_epoch and 'epoch' in df.columns:
                row = df[df['epoch'] == finish_epoch]
                if len(row) > 0:
                    return float(row.iloc[-1]['test_mean_pmae'])

            # 如果找不到 finish_epoch 对应行，就用 CSV 里的全局最小值。
            return float(df['test_mean_pmae'].min())
        except Exception as e:
            print(f"⚠️ 读取 baseline CSV 失败，使用 checkpoint min_avg：{e}")
            return baseline

    def _load_for_safe_finetune(self):
        print(f"📥 正在加载 epoch146 安全微调起点：{self.load_path}")
        checkpoint = torch.load(str(self.load_path), map_location=self.device)

        state_dict = self._remove_module_prefix(checkpoint['model'])
        state_dict = self.model.remap_legacy_state_dict(state_dict)
        load_result = self.model.load_state_dict(state_dict, strict=False)

        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)

        if len(missing_keys) > 0:
            print(f"ℹ️ strict=False 加载：缺失参数 {len(missing_keys)} 个，通常是新增 zero-init 模块")
            print("   missing keys 示例：", missing_keys[:8])
        if len(unexpected_keys) > 0:
            print(f"ℹ️ strict=False 加载：多余参数 {len(unexpected_keys)} 个")
            print("   unexpected keys 示例：", unexpected_keys[:8])

        self.C_net = self.model
        self.ema_model = AveragedModel(self.model)

        self.start_epoch = 1
        self.min_avg = self._infer_baseline_min_avg(checkpoint, self.load_path)
        self.best_epoch = 0

        # --load 微调不加载旧 optimizer。
        # v57/v59/v60 自动启用冻结式 adapter 微调；其他版本仍为全模型安全微调。
        self._apply_adapter_freeze()

        if not self.freeze_adapter_mode:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(self.args.beta1, self.args.beta2),
                weight_decay=self.args.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epoch,
                eta_min=1e-6
            )
            self.scheduler_mode = f"load_safe_long_cosine_Tmax_{self.args.epoch}"

        print(
            f"✅ 安全微调加载完成 | baseline_epoch={self.loaded_baseline_epoch} | "
            f"baseline_min_avg={self.min_avg:.6f} | 新优化器lr={self.args.lr:.8f} | "
            f"freeze_adapter={self.freeze_adapter_mode}"
        )


    def _set_requires_grad(self, module, flag):
        for param in module.parameters():
            param.requires_grad = flag

    def _apply_adapter_freeze(self):
        variant = getattr(self.args, 'exp_variant', 'v4')
        if not (self.safe_load_finetune and variant in self.freeze_adapter_variants):
            self.freeze_adapter_mode = False
            return

        self.freeze_adapter_mode = True

        # 冻结全部旧参数，包括 backbone / refinement / aggregation / Det2Sem / RCMA / 原始 head。
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_modules = []

        if variant in ['v57_v31_mass_allscale_residual', 'v60_v31_mass_allscale_residual_carb_wide']:
            if hasattr(self.model, 'mass_residual_correction'):
                self._set_requires_grad(self.model.mass_residual_correction, True)
                trainable_modules.append('mass_residual_correction')
            else:
                print("⚠️ 当前模型没有 mass_residual_correction，请检查 tsfr_net.py 与 exp_variant")

        if variant in ['v59_v31_carb_wide_adapter', 'v60_v31_mass_allscale_residual_carb_wide']:
            if hasattr(self.model, 'carb_residual_correction'):
                self._set_requires_grad(self.model.carb_residual_correction, True)
                trainable_modules.append('carb_residual_correction')
            else:
                print("⚠️ 当前模型没有 carb_residual_correction，请检查 tsfr_net.py 与 exp_variant")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in self.model.parameters())

        if len(trainable_params) == 0:
            raise RuntimeError("冻结式 adapter 微调失败：没有任何可训练参数")

        # 只把新增 adapter 参数交给优化器，避免旧参数被 Adam / weight decay 扰动。
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epoch,
            eta_min=1e-6
        )
        self.scheduler_mode = f"freeze_adapter_long_cosine_Tmax_{self.args.epoch}"

        print(
            f"🧊 已启用冻结式 adapter 微调 | variant={variant} | "
            f"trainable_modules={trainable_modules} | "
            f"trainable={trainable_count / 1e6:.4f}M / total={total_count / 1e6:.2f}M"
        )

    def _set_train_eval_mode(self):
        if not self.freeze_adapter_mode:
            self.model.train()
            self.C_net.train()
            return

        # 冻结主体使用 eval，防止 BN running stats 和 Dropout 改变 v31 的原始输出。
        self.model.eval()
        if hasattr(self.model, 'mass_residual_correction'):
            self.model.mass_residual_correction.train()
        if hasattr(self.model, 'carb_residual_correction'):
            self.model.carb_residual_correction.train()
        self.C_net = self.model


    def _resume_with_long_cosine(self):
        print(f"♻️ 正在从断点恢复训练：{self.resume_path}")
        checkpoint = torch.load(str(self.resume_path), map_location=self.device)

        state_dict = self._remove_module_prefix(checkpoint['model'])
        state_dict = self.model.remap_legacy_state_dict(state_dict)
        self.model.load_state_dict(state_dict)

        # 加载 optimizer 是为了保留 Adam 动量；随后强制重置 lr，避免沿用旧 checkpoint 的低 lr。
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr
            param_group['initial_lr'] = self.args.lr

        self.start_epoch = checkpoint.get('finish_epoch', 0) + 1
        self.min_avg = checkpoint.get('min_avg', self.min_avg)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        # 重新建立 scheduler，确保 base_lrs 使用当前 args.lr，而不是 checkpoint 里的旧 lr。
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epoch,
            eta_min=1e-6
        )
        self.scheduler_mode = f"resume_long_cosine_Tmax_{self.args.epoch}"

        # 对齐到已完成轮数。比如从 145 轮 best.pkl 续训，就把 lr 对齐到 cosine 的第145轮位置。
        completed_epoch = max(self.start_epoch - 1, 0)
        self._align_cosine_lr(completed_epoch)

        print(
            f"✅ 恢复成功 | 接续轮次：{self.start_epoch} | "
            f"best_epoch={self.best_epoch} | min_avg={self.min_avg:.6f} | "
            f"对齐后lr={self.optimizer.param_groups[0]['lr']:.8f}"
        )

    def _align_cosine_lr(self, completed_epoch):
        t_max = max(int(self.args.epoch), 1)
        eta_min = 1e-6
        completed_epoch = min(max(int(completed_epoch), 0), t_max)

        for param_group in self.optimizer.param_groups:
            base_lr = self.args.lr
            lr = eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * completed_epoch / t_max)) / 2.0
            param_group['lr'] = lr
            param_group['initial_lr'] = base_lr

        # 设置 scheduler 内部状态，使下一次 scheduler.step() 从当前轮次之后继续。
        self.scheduler.base_lrs = [self.args.lr for _ in self.optimizer.param_groups]
        self.scheduler.last_epoch = completed_epoch
        self.scheduler._step_count = completed_epoch + 1

    def _load_existing_history_for_resume(self):
        if not self.resume or not hasattr(self, 'csv_path') or not os.path.exists(self.csv_path):
            return

        try:
            old_df = pd.read_csv(self.csv_path)
            if 'epoch' in old_df.columns:
                old_df = old_df[old_df['epoch'] < self.start_epoch]
            for key in self.history.keys():
                if key in old_df.columns:
                    self.history[key] = old_df[key].tolist()
            print(f"📄 已读取旧训练日志，保留 {len(old_df)} 条历史记录，续写 training_history.csv")
        except Exception as e:
            print(f"⚠️ 读取旧训练日志失败，将重新写入 CSV：{e}")

    def _maybe_compile_for_training(self, epoch):
        if self.args.test:
            return

        if not self.gpu_mode:
            return

        if not self.enable_delayed_compile:
            return

        if self.freeze_adapter_mode:
            return

        if self.compile_failed or self.compiled_train_model:
            return

        if epoch <= self.compile_after_epoch:
            return

        if not hasattr(torch, "compile"):
            self.compile_failed = True
            print("⚠️ 当前 PyTorch 不支持 torch.compile，保持普通训练模式")
            return

        try:
            print(f"⚡ 第 {epoch} 轮开始启用 torch.compile 训练加速，本轮首次编译会稍慢")
            self.C_net = torch.compile(self.model, mode='reduce-overhead')
            self.compiled_train_model = True
            print("✅ torch.compile 训练模型已启用")
        except Exception as e:
            self.C_net = self.model
            self.compile_failed = True
            print(f"⚠️ torch.compile 启用失败，保持普通训练模式：{e}")

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    def train(self, epoch):
        self._maybe_compile_for_training(epoch)

        self._set_train_eval_mode()
        max_iter = len(self.train_data_loader)
        loss_avg = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for iter, x in enumerate(tqdm(self.train_data_loader, ncols=80)):
            inputs = x[0].to(self.device, non_blocking=True)
            total_calories = x[2].to(self.device, non_blocking=True).float()
            total_mass = x[3].to(self.device, non_blocking=True).float()
            total_fat = x[4].to(self.device, non_blocking=True).float()
            total_carb = x[5].to(self.device, non_blocking=True).float()
            total_protein = x[6].to(self.device, non_blocking=True).float()
            inputs_rgbd = x[7].to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                cal_pred, mass_pred, fat_pred, carb_pred, pro_pred = self.C_net(inputs, inputs_rgbd)

                calories_mae = self.L1Loss(cal_pred, total_calories)
                mass_mae = self.L1Loss(mass_pred, total_mass)
                fat_mae = self.L1Loss(fat_pred, total_fat)
                carb_mae = self.L1Loss(carb_pred, total_carb)
                protein_mae = self.L1Loss(pro_pred, total_protein)

                total_calories_loss = calories_mae / (total_calories.sum().item() + 1e-8)
                total_mass_loss = mass_mae / (total_mass.sum().item() + 1e-8)
                total_fat_loss = fat_mae / (total_fat.sum().item() + 1e-8)
                total_carb_loss = carb_mae / (total_carb.sum().item() + 1e-8)
                total_protein_loss = protein_mae / (total_protein.sum().item() + 1e-8)

                base_loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss
                loss = base_loss / self.accum_steps
                loss_avg += loss.item() * self.accum_steps

            if self.amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (iter + 1) % self.accum_steps == 0:
                if self.amp and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.ema_update_freq > 0 and iter % self.ema_update_freq == 0:
                self.ema_model.update_parameters(self.model)

        self.scheduler.step()
        train_loss = loss_avg / max_iter
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:2d} | Loss: {train_loss:.6f} | LR: {current_lr:.8f}')
        return train_loss, current_lr

    def test(self, epoch=0):
        # 测试默认走原始模型，避免 torch.compile 在 eval 阶段重新编译导致测试变慢。
        if self.fast_test_use_compiled_model and self.compiled_train_model:
            eval_model = self.C_net
        else:
            eval_model = self.model

        self.model.eval()
        eval_model.eval()

        calories_loss = mass_loss = fat_loss = carb_loss = protein_loss = 0.0
        calories_real = mass_real = fat_real = carb_real = protein_real = 0.0

        use_amp_test = self.gpu_mode and self.amp and self.fast_test_amp

        with torch.inference_mode():
            for x in tqdm(self.test_data_loader, ncols=80, desc="Testing"):
                inputs = x[0].to(self.device, non_blocking=True)
                total_calories = x[2].to(self.device, non_blocking=True).float().reshape(-1)
                total_mass = x[3].to(self.device, non_blocking=True).float().reshape(-1)
                total_fat = x[4].to(self.device, non_blocking=True).float().reshape(-1)
                total_carb = x[5].to(self.device, non_blocking=True).float().reshape(-1)
                total_protein = x[6].to(self.device, non_blocking=True).float().reshape(-1)
                inputs_rgbd = x[7].to(self.device, non_blocking=True)

                with autocast(enabled=use_amp_test):
                    cal_pred, mass_pred, fat_pred, carb_pred, pro_pred = eval_model(inputs, inputs_rgbd)

                cal_pred = cal_pred.float().reshape(-1)
                mass_pred = mass_pred.float().reshape(-1)
                fat_pred = fat_pred.float().reshape(-1)
                carb_pred = carb_pred.float().reshape(-1)
                pro_pred = pro_pred.float().reshape(-1)

                calories_loss += torch.abs(cal_pred - total_calories).sum().item()
                mass_loss += torch.abs(mass_pred - total_mass).sum().item()
                fat_loss += torch.abs(fat_pred - total_fat).sum().item()
                carb_loss += torch.abs(carb_pred - total_carb).sum().item()
                protein_loss += torch.abs(pro_pred - total_protein).sum().item()

                calories_real += total_calories.sum().item()
                mass_real += total_mass.sum().item()
                fat_real += total_fat.sum().item()
                carb_real += total_carb.sum().item()
                protein_real += total_protein.sum().item()

        calories_pmae = calories_loss / (calories_real + 1e-8)
        mass_pmae = mass_loss / (mass_real + 1e-8)
        fat_pmae = fat_loss / (fat_real + 1e-8)
        carb_pmae = carb_loss / (carb_real + 1e-8)
        protein_pmae = protein_loss / (protein_real + 1e-8)
        mean = (calories_pmae + mass_pmae + fat_pmae + carb_pmae + protein_pmae) / 5

        print(
            f'Test {epoch}: Mean={mean:.4f} | Cal={calories_pmae:.4f} '
            f'Mass={mass_pmae:.4f} Fat={fat_pmae:.4f} '
            f'Carb={carb_pmae:.4f} Pro={protein_pmae:.4f}'
        )

        # 安全微调：只有超过 baseline / 当前 best 才覆盖 best.pkl。
        if not self.args.test and mean < self.min_avg:
            self.min_avg = mean
            self.best_epoch = epoch
            self.save(epoch)
            print(f"✅ 最优模型已保存！")

        return mean, calories_pmae, mass_pmae, fat_pmae, carb_pmae, protein_pmae

    def main(self):
        if self.args.test:
            self.load_test(self.load_path)
            self.test(0)
            return

        print(
            f"🚀 TSFR-Net训练启动 | exp_variant={getattr(self.args, 'exp_variant', 'v4')} | "
            f"目标总轮数={self.epoch} | LR策略={self.scheduler_mode}"
        )

        for epoch in range(self.start_epoch, self.epoch + 1):
            train_loss, current_lr = self.train(epoch)
            mean_pmae, cal, mass, fat, carb, pro = self.test(epoch)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["test_mean_pmae"].append(mean_pmae)
            self.history["cal_pmae"].append(cal)
            self.history["mass_pmae"].append(mass)
            self.history["fat_pmae"].append(fat)
            self.history["carb_pmae"].append(carb)
            self.history["protein_pmae"].append(pro)
            self.history["lr"].append(current_lr)

            pd.DataFrame(self.history).to_csv(self.csv_path, index=False)

            if self.tb_writer:
                self.tb_writer.add_scalar('Train/Loss', train_loss, epoch)
                self.tb_writer.add_scalar('Test/Mean_PMAE', mean_pmae, epoch)
                self.tb_writer.add_scalar('Test/Cal_PMAE', cal, epoch)
                self.tb_writer.add_scalar('Test/Mass_PMAE', mass, epoch)
                self.tb_writer.add_scalar('Test/Fat_PMAE', fat, epoch)
                self.tb_writer.add_scalar('Test/Carb_PMAE', carb, epoch)
                self.tb_writer.add_scalar('Test/Protein_PMAE', pro, epoch)
                self.tb_writer.add_scalar('Optimizer/Learning_Rate', current_lr, epoch)
                if epoch % 3 == 0:
                    self.tb_writer.flush()

        if self.tb_writer:
            self.tb_writer.close()

        print(f"\n🎉 训练完成！最佳PMAE: {self.min_avg:.4f} | 最优轮数: {self.best_epoch}")

    def save(self, save_epoch):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'finish_epoch': save_epoch,
            'min_avg': self.min_avg,
            'best_epoch': self.best_epoch,
            'scheduler_mode': self.scheduler_mode,
            'exp_variant': getattr(self.args, 'exp_variant', 'v4'),
            'loaded_from': str(self.load_path) if self.safe_load_finetune else '',
            'loaded_baseline_epoch': self.loaded_baseline_epoch
        }
        torch.save(save_dict, os.path.join(self.dump_path, 'best.pkl'), _use_new_zipfile_serialization=False)

    def load_test(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        state_dict = self._remove_module_prefix(checkpoint['model'])
        state_dict = self.model.remap_legacy_state_dict(state_dict)
        load_result = self.model.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) > 0:
            print(f"⚠️ 测试加载缺失参数 {len(load_result.missing_keys)} 个：", list(load_result.missing_keys)[:8])
        if len(load_result.unexpected_keys) > 0:
            print(f"⚠️ 测试加载多余参数 {len(load_result.unexpected_keys)} 个：", list(load_result.unexpected_keys)[:8])
