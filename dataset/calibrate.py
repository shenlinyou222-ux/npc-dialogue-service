# -*- coding: utf-8 -*-
"""
calibrate.py —— 长度标定：在"合并轮数上限 cap"和"单行补量 S"两个旋钮上搜索，
让最终数据集的 assistant 平均长度落进父代理给的 20-30 token 目标区间，
同时样本数保持 800-1500。同时计算 gold（真实原句）的长度分布用于对比。
"""
import collections, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import (build_turns, find_runs, iter_jsonl, CORPUS, CHAR,
                           load_tokenizer)
import glob

BM = os.path.join(CORPUS, "by_mission")
GOLD = "/mnt/e/habitica-setup/job/gold-compare.json"
ntok, _ = load_tokenizer("/home/user/npc-models/MiniCPM5-1B")


def pct(v, p):
    if not v:
        return 0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def q(v):
    v = sorted(v)
    return (v[0], pct(v, .5), pct(v, .9), v[-1], round(statistics.mean(v), 1))


# ---------------- 1) gold 参考 ----------------
print("=" * 88); print("1) gold（真实原句）长度分布"); print("=" * 88)
g = json.load(open(GOLD, encoding="utf-8"))
recs = g["records"]
gold_chars = [len(r["gold"]) for r in recs]
gold_tok = [ntok(r["gold"]) for r in recs]
print(f"gold 条数={len(recs)}")
print(f"  char : min={min(gold_chars)} p50={pct(gold_chars,.5)} p90={pct(gold_chars,.9)} max={max(gold_chars)} mean={statistics.mean(gold_chars):.1f}")
print(f"  token: min={min(gold_tok)} p50={pct(gold_tok,.5)} p90={pct(gold_tok,.9)} max={max(gold_tok)} mean={statistics.mean(gold_tok):.1f}")
print(f"  每条 gold: {[ (len(r['gold']), r['gold']) for r in recs[:3]]}")
lora_out = [r["runs"]["lora"]["output"] for r in recs]
base_out = [r["runs"]["base"]["output"] for r in recs]
print(f"  LoRA 输出 char mean={statistics.mean(len(x) for x in lora_out):.1f}  token mean={statistics.mean(ntok(x) for x in lora_out):.1f}")
print(f"  基座 输出 char mean={statistics.mean(len(x) for x in base_out):.1f}  token mean={statistics.mean(ntok(x) for x in base_out):.1f}")

# ---------------- 2) 原始游程统计 ----------------
print(); print("=" * 88); print("2) 三月七发言游程（每个游程 = 一段可作 assistant 的连续台词）"); print("=" * 88)
runs = []
for fp in sorted(glob.glob(os.path.join(BM, "*.jsonl"))):
    rows = list(iter_jsonl(fp))
    turns = build_turns(rows)
    for (s, e, texts) in find_runs(turns):
        texts = [t.strip() for t in texts if t.strip()]
        if texts:
            runs.append((os.path.basename(fp)[:-6], texts))
print("游程数:", len(runs))
byl = collections.Counter(len(t) for _, t in runs)
print("游程行数分布:", dict(sorted(byl.items())))
for n in sorted(byl):
    sub = ["\n".join(t) for _, t in runs if len(t) == n]
    ch = [len(x) for x in sub]; tk = [ntok(x) for x in sub]
    print(f"  {n} 行: n={len(sub):5d}  char mean={statistics.mean(ch):6.1f}  token mean={statistics.mean(tk):5.1f}")

# ---------------- 3) cap 切块 + 单行补量 搜索 ----------------
print(); print("=" * 88); print("3) 策略搜索：cap（每个样本最多合并几轮）× S（补充的长单行条数）"); print("=" * 88)
multi_all = [(m, t) for m, t in runs if len(t) >= 2]
singles = [t[0] for _, t in runs if len(t) == 1]
singles_long = sorted([s for s in singles if len(s) >= 20], key=lambda s: -ntok(s))
print(f"多行游程(>=2行): {len(multi_all)}   单行游程: {len(singles)}  其中>=20字: {len(singles_long)}")

for cap in (2, 3, 4):
    chunks = []
    for m, t in multi_all:
        for i in range(0, len(t), cap):
            piece = "\n".join(t[i:i + cap])
            if ntok(piece) >= 8:
                chunks.append((m, piece))
    ck_tok = [ntok(p) for _, p in chunks]
    ck_ch = [len(p) for _, p in chunks]
    print(f"\ncap={cap}: 切块后样本 {len(chunks)} 条  token mean={statistics.mean(ck_tok):.1f} "
          f"char mean={statistics.mean(ck_ch):.1f}")
    for S in (0, 200, 400, 600):
        sub = ck_tok + [ntok(s) for s in singles_long[:S]]
        subc = ck_ch + [len(s) for s in singles_long[:S]]
        n = len(sub)
        print(f"    +单行{S:4d} -> n={n:5d}  token: mean={statistics.mean(sub):5.1f} "
              f"p50={pct(sub,.5):3d} p90={pct(sub,.9):3d} min={min(sub):3d} max={max(sub):3d} "
              f"| char mean={statistics.mean(subc):5.1f}")

# ---------------- 4) 推荐点 ----------------
print(); print("=" * 88); print("4) 命中最优区间的组合（目标 token mean 20-30，n 800-1500）"); print("=" * 88)
best = []
for cap in (2, 3, 4):
    chunks = []
    for m, t in multi_all:
        for i in range(0, len(t), cap):
            piece = "\n".join(t[i:i + cap])
            if ntok(piece) >= 8:
                chunks.append(ntok(piece))
    for S in range(0, len(singles_long) + 1, 25):
        sub = chunks + [ntok(s) for s in singles_long[:S]]
        n = len(sub)
        mu = statistics.mean(sub)
        if 800 <= n <= 1500 and 20 <= mu <= 30:
            best.append((abs(mu - 27), cap, S, n, round(mu, 2), pct(sub, .5), pct(sub, .9)))
best.sort()
for b in best[:15]:
    print(f"  cap={b[1]} S={b[2]:4d} -> n={b[3]:5d} token mean={b[4]:5.1f} p50={b[5]} p90={b[6]}")
if not best:
    print("  （无组合落在目标区间，需放宽 cap 上限或增加单行阈值）")
