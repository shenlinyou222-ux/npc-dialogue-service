# -*- coding: utf-8 -*-
"""准备 HSR LoRA 训练数据。

关键设计：
1. **按 mission 切分 train/val** —— 同一任务内上下文高度重叠，随机切会数据泄漏。
   本脚本以 mission 为最小不可分割单元做分组切分，保证验证集任务在训练集中完全不出现。
2. **角色平衡** —— 开拓者（玩家角色）占 19%，会主导语气。对其做配额下采样，
   保证 53 个主要角色都被看到，同时不让单一角色吃掉训练信号。
3. **保留 kind=="choice"** —— 任务二已说明目标是推测解码（前缀树+专有名词表）的接受率，
   需要模型记住专有名词与选项式措辞，故不做过滤；统计仍然输出，便于事后裁剪。
4. 输出 chat 格式 messages，由训练脚本用模型自带 chat_template 渲染。
"""
import argparse
import collections
import json
import random
from pathlib import Path

SRC = Path("/mnt/c/Users/user/Desktop/ds harness/game_dialogue_corpus_wiki/lora_samples.jsonl")
OUT = Path("/mnt/c/Users/user/Desktop/ds harness/lora_hsr/data")


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] 第 {ln} 行解析失败: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--mc-cap", type=int, default=12000,
                    help="开拓者(玩家角色)配额上限，0=不限")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.src)
    print(f"读入 {len(rows)} 条")

    # ---------- 基本校验 ----------
    bad = 0
    clean = []
    for r in rows:
        msgs = r.get("messages") or []
        if len(msgs) < 3 or msgs[-1].get("role") != "assistant" or not msgs[-1].get("content"):
            bad += 1
            continue
        if not r.get("mission"):
            r["mission"] = "__no_mission__"
        clean.append(r)
    print(f"结构异常丢弃 {bad} 条，保留 {len(clean)} 条")

    # ---------- 按 mission 分组 ----------
    by_mission = collections.defaultdict(list)
    for r in clean:
        by_mission[r["mission"]].append(r)
    missions = sorted(by_mission.keys())
    print(f"mission 数: {len(missions)}")

    rng = random.Random(args.seed)
    rng.shuffle(missions)

    target_val = int(len(clean) * args.val_ratio)
    val_m, cur = [], 0
    for m in missions:
        if cur >= target_val:
            break
        val_m.append(m)
        cur += len(by_mission[m])
    val_set = set(val_m)
    train_m = [m for m in missions if m not in val_set]

    train_rows = [r for m in train_m for r in by_mission[m]]
    val_rows = [r for m in val_m for r in by_mission[m]]

    # ---------- 角色配额下采样（只作用于 train） ----------
    capped_note = "未启用"
    if args.mc_cap > 0:
        mc = "开拓者"
        mc_rows = [r for r in train_rows if r.get("character") == mc]
        other = [r for r in train_rows if r.get("character") != mc]
        if len(mc_rows) > args.mc_cap:
            rng.shuffle(mc_rows)
            mc_rows = mc_rows[:args.mc_cap]
            capped_note = f"{mc} 限制在 {args.mc_cap} 条"
        train_rows = mc_rows + other
        rng.shuffle(train_rows)

    # ---------- 统计 ----------
    def stats(rs):
        cc = collections.Counter(r.get("character") for r in rs)
        return {
            "n": len(rs),
            "n_missions": len(set(r["mission"] for r in rs)),
            "task": dict(collections.Counter(r.get("task") for r in rs)),
            "kind": dict(collections.Counter(r.get("kind") for r in rs)),
            "char_top20": cc.most_common(20),
            "n_characters": len(cc),
        }

    st = {
        "src": str(args.src),
        "seed": args.seed,
        "val_ratio_target": args.val_ratio,
        "mission_cap_note": capped_note,
        "split": "by_mission(no leakage)",
        "train": stats(train_rows),
        "val": stats(val_rows),
        "leakage_check": {
            "mission_overlap": len(set(r["mission"] for r in train_rows) &
                                    set(r["mission"] for r in val_rows)),
        },
    }

    def dump(path, rs):
        rs = list(rs)
        rng.shuffle(rs)
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写出 {path} : {len(rs)} 条")

    dump(out / "train.jsonl", train_rows)
    dump(out / "val.jsonl", val_rows)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

    print("\n=== 切分结果 ===")
    print(f"train: {st['train']['n']} 条 / {st['train']['n_missions']} 个 mission")
    print(f"val  : {st['val']['n']} 条 / {st['val']['n_missions']} 个 mission")
    print(f"mission 重叠(必须为0): {st['leakage_check']['mission_overlap']}")
    print(f"角色配额: {capped_note}")
    print("train 角色 top10:", st["train"]["char_top20"][:10])
    print("train kind:", st["train"]["kind"])
    print("val   kind:", st["val"]["kind"])


if __name__ == "__main__":
    main()
