# -*- coding: utf-8 -*-
"""HSR 角色对话 LoRA 微调 Qwen3.5-0.8B。

关键点（踩过的坑，逐条对应）：
1. 基座 arch = Qwen3_5ForConditionalGeneration（多模态 VL）。**必须用这个 arch 训练**，
   因为 AutoModelForCausalLM 会加载 text-only 的 Qwen3_5ForCausalLM，模块路径是
   `model.layers.*`，而全量模型的适配器路径是 `model.language_model.layers.*` —— 前者存出来的
   adapter 在 vLLM 里对不上 key 会静默不生效。已实测 adapter key 形如
   `base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight`。
2. 该模型是 **混合注意力**：24 层里 18 层 linear_attn + 6 层 full self_attn。
   只盯 q/k/v/o_proj 会漏掉 3/4 的层，所以 target 里额外纳入 linear_attn 的
   in_proj_qkv / in_proj_z / out_proj。**排除 conv1d**（裸 torch Conv1d，无 in/out_features，
   PEFT 会把它当 Linear 处理；且 linear_attn.out_proj 与 self_attn.o_proj 维度不同
   —— 2048->1024 vs 2048->1024 但 in 维不一致），因此下面用「推断 in_features/out_features
   相等且 >= 128」的方式筛选，避免 PEFT 假设方阵权重导致形状错误。
3. 只对 assistant 段算 loss（prompt 段 -100）。实测该 tokenizer 的 chat_template 会给
   assistant 轮注入 `<think>\n\n</think>\n\n`，这部分落在 prompt 里被掩掉，模型不学 thinking。
4. 梯度检查点 + 分段 checkpoint，防中断丢进度。
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_MODEL = "/home/user/npc-models/Qwen3.5-0.8B"
DEFAULT_DATA = "/mnt/c/Users/user/Desktop/ds harness/lora_hsr/data"
DEFAULT_OUT = "/home/user/npc-models/lora-hsr"


def log(msg):
    print(msg, flush=True)


def infer_io(mod):
    """返回 (in_features, out_features)；非二维权重返回 None。"""
    ins = getattr(mod, "in_features", None)
    outs = getattr(mod, "out_features", None)
    if isinstance(ins, int) and isinstance(outs, int):
        return ins, outs
    # peft 包装过的 module.torch.nn.Linear 可能被藏在子模块里
    for _, sub in mod.named_children():
        w = getattr(sub, "weight", None)
        if hasattr(w, "dim") and w.dim() == 2:
            return w.shape[1], w.shape[0]
    w = getattr(mod, "weight", None)
    if w is not None and hasattr(w, "dim") and w.dim() == 2:
        # 裸 Conv1d 也走这里；排除掉
        if isinstance(mod, torch.nn.Conv1d):
            return None
        return w.shape[1], w.shape[0]
    return None


CANDIDATE_TAILS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "out_proj")


def pick_targets(model, min_dim=16):
    """按模块尾部名 + 推断维度筛选 LoRA 目标。返回 (targets, detail)。

    实测维度：q 1024->4096, k/v 1024->512, o 2048->1024, gate/up 1024->3584,
    down 3584->1024, linear_attn.in_proj_qkv 1024->6144, out_proj 2048->1024,
    in_proj_z 1024->2048, in_proj_a/b 1024->16（门控，维度太小，min_dim 排除）。
    """
    found = {}   # tail -> (count, io)
    for name, mod in model.named_modules():
        tail = name.split(".")[-1]
        if tail not in CANDIDATE_TAILS:
            continue
        io = infer_io(mod)
        if io is None:
            continue
        ins, outs = io
        if ins < min_dim or outs < min_dim:
            continue
        if tail in found:
            found[tail] = (found[tail][0] + 1, found[tail][1])
        else:
            found[tail] = (1, io)
    targets = sorted(found.keys())
    return targets, found


class SFTData(Dataset):
    """渲染 chat 模板，prompt 段打 -100。整条序列为一个样本。"""

    def __init__(self, rows, tok, max_len=512, verbose=False):
        self.items = []
        self.skipped = 0
        self.lens = []
        self.prefix_mismatch = 0
        for r in rows:
            msgs = r["messages"]
            def render(m, gen):
                try:
                    text = tok.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=gen,
                        enable_thinking=False)
                except TypeError:
                    text = tok.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=gen)
                return tok.encode(text, add_special_tokens=False)

            prompt_ids = render(msgs[:-1], True)
            full_ids = render(msgs, False)
            if full_ids and full_ids[-1] != tok.eos_token_id:
                full_ids = full_ids + [tok.eos_token_id]
            if len(full_ids) > max_len:
                self.skipped += 1
                continue
            if len(prompt_ids) > len(full_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
                # prompt 不是 full 的前缀（模板行为异常）→ 退化处理，只在末尾对齐
                self.prefix_mismatch += 1
                prompt_ids = full_ids[:-1]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            if all(l == -100 for l in labels):
                self.skipped += 1
                continue
            self.items.append((full_ids, labels))
            self.lens.append(len(full_ids))
        if verbose:
            ls = sorted(self.lens)
            if ls:
                log(f"  长度分位: min={ls[0]} p50={ls[len(ls)//2]} "
                    f"p95={ls[int(len(ls)*0.95)]} max={ls[-1]}  "
                    f"截断丢弃={self.skipped} 前缀不匹配={self.prefix_mismatch}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    ml = max(len(ids) for ids, _ in batch)
    ii, lb, am = [], [], []
    for ids, lab in batch:
        pad = ml - len(ids)
        ii.append(ids + [pad_id] * pad)
        lb.append(lab + [-100] * pad)
        am.append([1] * len(ids) + [0] * pad)
    return torch.tensor(ii), torch.tensor(lb), torch.tensor(am)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=0, help="0 => 2*r，对齐参考配置比例")
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="训练集截断条数，0=全量")
    ap.add_argument("--val-limit", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=0, help="0=按 epochs 跑满")
    ap.add_argument("--ckpt-every", type=int, default=300, help="每多少优化步存 checkpoint")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="只做数据/模型自检后退出")
    ap.add_argument("--no-ckpt", action="store_true", help="关闭梯度检查点（更吃显存）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.alpha == 0:
        args.alpha = 2 * args.r

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    log(f"=== 加载 tokenizer/model: {args.model}")
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa")
    log(f"    arch = {type(model).__name__}   (必须是 Qwen3_5ForConditionalGeneration)")

    targets, detail = pick_targets(model)
    log(f"=== LoRA target_modules ({len(targets)}): {targets}")
    for t, (n, io) in sorted(detail.items()):
        log(f"      {t:12s} x{n:<3d} io={io}")

    # 冻结视觉塔，省显存且与任务无关
    vt = 0
    for n, p in model.named_parameters():
        if n.startswith("model.visual") or ".visual." in n:
            p.requires_grad_(False)
            vt += p.numel()
    log(f"    冻结视觉塔参数: {vt/1e6:.1f}M")

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    log("=== 读取数据")
    def load(p):
        if not Path(p).exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    tr_rows = load(Path(args.data) / "train.jsonl")
    va_rows = load(Path(args.data) / "val.jsonl")
    rng = random.Random(args.seed)
    rng.shuffle(tr_rows)
    if args.limit:
        tr_rows = tr_rows[:args.limit]
    log(f"  train 原始 {len(tr_rows)}  val 原始 {len(va_rows)}")
    train = SFTData(tr_rows, tok, args.maxlen, verbose=True)
    if args.val_limit and len(va_rows) > args.val_limit:
        va_rows = rng.sample(va_rows, args.val_limit)
    val = SFTData(va_rows, tok, args.maxlen, verbose=True)
    log(f"  编码后 train={len(train)}  val={len(val)}")

    if args.dry_run:
        log("dry-run 结束（未训练）")
        return

    model.cuda()
    model.config.use_cache = False
    if not args.no_ckpt:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        log("    梯度检查点: 已开启")
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    pad_id = tok.pad_token_id or tok.eos_token_id
    dl = DataLoader(train, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id), drop_last=False)
    vdl = DataLoader(val, batch_size=args.bs, shuffle=False,
                     collate_fn=lambda b: collate(b, pad_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps_per_ep = math.ceil(len(dl) / args.accum)
    total_steps = steps_per_ep * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.03)
    log(f"=== 开始训练: {len(dl)} batch/epoch, accum={args.accum}, "
        f"{steps_per_ep} 优化步/epoch, total_steps={total_steps}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    stop = False
    hist = []
    for ep in range(args.epochs):
        if stop:
            break
        model.train()
        for i, (ids, labels, attn) in enumerate(dl):
            ids, labels, attn = ids.cuda(), labels.cuda(), attn.cuda()
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / args.accum).backward()
            run_loss += out.loss.item()
            run_n += 1
            if (i + 1) % args.accum == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                if step < total_steps - 1:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == 1:
                    dt = time.time() - t0
                    rec = {"step": step, "epoch": ep + 1, "loss": run_loss / max(run_n, 1),
                           "lr": sched.get_last_lr()[0], "sec_per_step": dt / step}
                    hist.append(rec)
                    log(f"ep{ep+1} step {step}/{total_steps} loss={rec['loss']:.4f} "
                        f"lr={rec['lr']:.2e} {rec['sec_per_step']:.2f}s/step")
                    run_loss, run_n = 0.0, 0
                if args.ckpt_every and step % args.ckpt_every == 0:
                    ck = outdir / f"checkpoint-{step}"
                    model.save_pretrained(str(ck))
                    log(f"  [ckpt] 已存 {ck}")
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
        if stop:
            break

        # ---- 每 epoch 末评估 ----
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for ids, labels, attn in vdl:
                o = model(input_ids=ids.cuda(), attention_mask=attn.cuda(),
                          labels=labels.cuda())
                vl += o.loss.item()
                vn += 1
        model.train()
        vloss = vl / max(vn, 1)
        hist.append({"step": step, "epoch": ep + 1, "val_loss": vloss})
        log(f"[epoch {ep+1} 结束] val_loss={vloss:.4f}  用时 {(time.time()-t0)/60:.1f} min")
        model.save_pretrained(str(outdir / f"checkpoint-ep{ep+1}"))
        log(f"  [ckpt] 已存 {outdir}/checkpoint-ep{ep+1}")

    model.save_pretrained(str(outdir))
    tok.save_pretrained(str(outdir))
    with open(outdir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)
    log(f"=== 完成: {step} 步, 总耗时 {(time.time()-t0)/60:.1f} min")
    log(f"adapter 已保存: {outdir}")


if __name__ == "__main__":
    main()
