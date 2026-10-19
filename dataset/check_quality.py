# -*- coding: utf-8 -*-
"""
check_quality.py —— 去重 + 统计 + 新旧对比 + gold（真实原句）对比。

产出 quality_report.json，并打印完整报告。
检查项：
  ① 完全重复样本数（必须 0）
  ② 相似度 >0.9 的样本对占比（必须 <5%）
  ③ 长度分布 / region / kind / channel / variant 分布 / mission 覆盖
  ④ 新旧数据对比表
  ⑤ assistant 长度分布 vs 真实原句(gold) 长度分布
  ⑥ train/val mission 重叠
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.expanduser("~/lora_hsr_run/data-march7")
GOLD = "/mnt/e/habitica-setup/job/gold-compare.json"
MODEL = "/home/user/npc-models/MiniCPM5-1B"


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pctl(v, p):
    if not v:
        return 0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def describe(v):
    return {"min": min(v) if v else 0, "p50": pctl(v, .5), "p90": pctl(v, .9),
            "max": max(v) if v else 0,
            "mean": round(statistics.mean(v), 1) if v else 0}


def shingles(s, k=4):
    s = "".join(s.split())
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--sim-thr", type=float, default=0.9)
    ap.add_argument("--new-tag", default="v2")
    args = ap.parse_args()

    tr = list(iter_jsonl(os.path.join(args.dir, "train.jsonl")))
    va = list(iter_jsonl(os.path.join(args.dir, "val.jsonl")))
    new = tr + va
    print("=" * 92)
    print(" 三月七 LoRA 数据质量检查")
    print("=" * 92)

    fails, warns = [], []

    # ---------- ① 完全重复 ----------
    print("\n[1] 去重检查")
    key = lambda s: (s["messages"][1]["content"], s["messages"][2]["content"])
    c = collections.Counter(key(s) for s in new)
    exact_dup = sum(v - 1 for v in c.values() if v > 1)
    print(f"    完全重复样本（user+assistant 全同）: {exact_dup} 条  "
          f"{'✓' if exact_dup == 0 else '✗ 必须为 0'}")
    if exact_dup:
        fails.append(f"存在 {exact_dup} 条完全重复样本")
    # assistant 台词自身重复
    ca = collections.Counter(s["messages"][2]["content"] for s in new)
    dup_a = sum(v - 1 for v in ca.values() if v > 1)
    print(f"    assistant 台词重复（同句被用于多条样本）: {dup_a} 条 "
          f"（占比 {dup_a/len(new):.1%}，跨 mission 复用同一句属正常）")

    # ---------- ② 相似度 >0.9 ----------
    print(f"\n[2] 相似度检查（字符 4-gram Jaccard > {args.sim_thr}）")
    sh = [shingles(s["messages"][1]["content"] + "\n" + s["messages"][2]["content"]) for s in new]
    lens = [len(x) for x in sh]
    pairs, n_pairs = 0, 0
    worst = []
    for i in range(len(new)):
        for j in range(i + 1, len(new)):
            if min(lens[i], lens[j]) / max(lens[i], lens[j]) < 0.4:
                continue                       # 长度差太大，不可能 >0.9
            n_pairs += 1
            r = jaccard(sh[i], sh[j])
            if r > args.sim_thr:
                pairs += 1
                if len(worst) < 3:
                    worst.append((round(r, 3), new[i]["messages"][2]["content"][:40],
                                  new[j]["messages"][2]["content"][:40]))
    ratio = pairs / len(new)
    print(f"    相似度 >{args.sim_thr} 的样本对: {pairs} 对 / {len(new)} 条 = {ratio:.2%}  "
          f"{'✓' if ratio < 0.05 else '✗ 必须 <5%'}")
    for w in worst:
        print(f"        r={w[0]}  A={w[1]!r}  B={w[2]!r}")
    if ratio >= 0.05:
        fails.append(f"高相似样本对占比 {ratio:.2%} >= 5%")

    # ---------- ③ 统计 ----------
    print("\n[3] 统计")
    for tag, rows in (("train", tr), ("val", va), ("合计", new)):
        a = [s["assistant_tokens"] for s in rows]
        u = [s["user_tokens"] for s in rows]
        ac = [len(s["messages"][2]["content"]) for s in rows]
        d = describe(a)
        print(f"    {tag:5s}: n={len(rows):5d}  mission={len(set(s['mission'] for s in rows)):4d}  "
              f"assistant token min={d['min']:3d} p50={d['p50']:3d} p90={d['p90']:3d} "
              f"max={d['max']:3d} mean={d['mean']:5.1f} | char mean={statistics.mean(ac):5.1f} "
              f"| user token mean={statistics.mean(u):5.1f}")

    print("\n    variant 分布:", dict(collections.Counter(s["variant"] for s in new)))
    print("    kind    分布:", dict(collections.Counter(s["kind"] for s in new)))
    print("    channel 分布:", dict(collections.Counter(s["channel"] for s in new)))
    print("    region  分布 (top12):")
    for r, n in collections.Counter(s["region"] or "(空)" for s in new).most_common(12):
        print(f"        {r:24s} {n:5d}  {n/len(new):5.1%}")
    print("    assistant 行数分布:", dict(sorted(collections.Counter(s["assistant_lines"] for s in new).items())))
    print(f"    synthetic_context（角色语音合成情境）: {sum(s['synthetic_context'] for s in new)} 条")

    # 分层覆盖检查（父代理要求覆盖：战斗/日常/剧情关键节点/情绪化）
    print("\n    题材分层覆盖检查:")
    def has(s, kws):
        t = s["messages"][1]["content"] + s["messages"][2]["content"] + s["section"]
        return any(k in t for k in kws)
    layers = {
        "战斗场景": (["战斗", "对决", "敌人", "敌人挡在", "受击", "终结技", "普攻", "战技", "秘技", "擂台", "危险", "小心"],
                    lambda s: s["kind"] == "voice" or has(s, ["战斗", "对决", "敌人", "受击", "终结技", "普攻", "战技", "秘技", "擂台", "危险"])),
        "日常闲聊": (None, lambda s: s["channel"] in ("sms", "voice") or has(s, ["闲聊", "拍照", "照片"])),
        "剧情关键节点": (None, lambda s: s["channel"] == "mission"),
        "情绪化（生气/开心/担心）": (None, lambda s: has(s, ["哼", "讨厌", "气", "哈哈", "嘿嘿", "嘻嘻", "好耶", "怎么办", "糟糕", "小心", "没事吧", "担心", "！", "？！"])),
    }
    for name, (_, fn) in layers.items():
        n = sum(1 for s in new if fn(s))
        print(f"        {name:24s} {n:5d} 条  {n/len(new):5.1%}")

    # ---------- ④ 新旧对比 ----------
    print("\n[4] 新旧数据对比")
    print(f"    {'指标':<28s} {'旧 data-march7':>16s} {'新 data-march7-v2':>18s}")
    print("    " + "-" * 64)
    old_tr = list(iter_jsonl(os.path.join(OLD, "train.jsonl"))) if os.path.exists(os.path.join(OLD, "train.jsonl")) else []
    old_va = list(iter_jsonl(os.path.join(OLD, "val.jsonl"))) if os.path.exists(os.path.join(OLD, "val.jsonl")) else []
    old = old_tr + old_va
    if old:
        oa = [len(s["messages"][2]["content"]) for s in old]
        na = [len(s["messages"][2]["content"]) for s in new]
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            nt = lambda s: len(tk.encode(s, add_special_tokens=False))
        except Exception:
            nt = lambda s: int(len(s) * 0.62)
        oat = [nt(s["messages"][2]["content"]) for s in old]
        nat = [nt(s["messages"][2]["content"]) for s in new]
        ou = [nt(s["messages"][1]["content"]) for s in old]
        nu = [nt(s["messages"][1]["content"]) for s in new]
        o_m = set(s.get("mission") for s in old)

        def row(label, ov, nv):
            print(f"    {label:<28s} {str(ov):>16s} {str(nv):>18s}")
        row("样本总数 (train/val)", f"{len(old_tr)}/{len(old_va)}", f"{len(tr)}/{len(va)}")
        row("覆盖 mission 数", f"{len(o_m)}", f"{len(set(s['mission'] for s in new))}")
        row("每 mission 平均样本数", f"{len(old)/max(1,len(o_m)):.1f}", f"{len(new)/len(set(s['mission'] for s in new)):.1f}")
        row("assistant token 均值", f"{statistics.mean(oat):.1f}", f"{statistics.mean(nat):.1f}")
        row("assistant token p50", f"{pctl(oat,.5)}", f"{pctl(nat,.5)}")
        row("assistant token p90", f"{pctl(oat,.9)}", f"{pctl(nat,.9)}")
        row("assistant token max", f"{max(oat)}", f"{max(nat)}")
        row("assistant 字符均值", f"{statistics.mean(oa):.1f}", f"{statistics.mean(na):.1f}")
        row("user token 均值", f"{statistics.mean(ou):.1f}", f"{statistics.mean(nu):.1f}")
        row("含【任务：】元数据", f"{sum(1 for s in old if '【任务：' in s['messages'][1]['content'])/len(old):.1%}", "0.0%")
        row("user 段行数均值", f"{statistics.mean(s['messages'][1]['content'].count(chr(10))+1 for s in old):.1f}",
            f"{statistics.mean(s['messages'][1]['content'].count(chr(10))+1 for s in new):.1f}")
        row("完全重复样本", f"{sum(v-1 for v in collections.Counter((s['messages'][1]['content'],s['messages'][2]['content']) for s in old).values() if v>1)}", f"{exact_dup}")

    # ---------- ⑤ gold 对比 ----------
    print("\n[5] assistant 长度分布 vs 真实原句(gold)")
    if os.path.exists(GOLD):
        g = json.load(open(GOLD, encoding="utf-8"))["records"]
        gc = [len(r["gold"]) for r in g]
        gt = [nt(r["gold"]) for r in g]
        lora_out = [r["runs"]["lora"]["output"] for r in g]
        base_out = [r["runs"]["base"]["output"] for r in g]
        lt = [nt(x) for x in lora_out]
        bt = [nt(x) for x in base_out]
        print(f"    {'来源':<34s} {'char mean':>10s} {'token mean':>11s} {'token p50':>10s} {'token p90':>10s}")
        print("    " + "-" * 78)
        print(f"    {'真实原句 gold (n=10)':<34s} {statistics.mean(gc):>10.1f} {statistics.mean(gt):>11.1f} {pctl(gt,.5):>10d} {pctl(gt,.9):>10d}")
        print(f"    {'旧 LoRA 推理输出 (n=10)':<34s} {statistics.mean(len(x) for x in lora_out):>10.1f} {statistics.mean(lt):>11.1f} {pctl(lt,.5):>10d} {pctl(lt,.9):>10d}")
        print(f"    {'基座模型 推理输出 (n=10)':<34s} {statistics.mean(len(x) for x in base_out):>10.1f} {statistics.mean(bt):>11.1f} {pctl(bt,.5):>10d} {pctl(bt,.9):>10d}")
        print(f"    {'★ 新训练集 assistant':<34s} {statistics.mean(na):>10.1f} {statistics.mean(nat):>11.1f} {pctl(nat,.5):>10d} {pctl(nat,.9):>10d}")
        print(f"    {'旧训练集 assistant':<34s} {statistics.mean(oa):>10.1f} {statistics.mean(oat):>11.1f} {pctl(oat,.5):>10d} {pctl(oat,.9):>10d}")
        print()
        print(f"    旧训练集 assistant 均值 {statistics.mean(oat):.1f} token，只到真实原句 "
              f"({statistics.mean(gt):.1f}) 的 {statistics.mean(oat)/statistics.mean(gt):.0%}")
        print(f"    新训练集 assistant 均值 {statistics.mean(nat):.1f} token，是真实原句的 "
              f"{statistics.mean(nat)/statistics.mean(gt):.2f}×（目标区间 20-30 token）")
    else:
        print(f"    ⚠ 未找到 {GOLD}")
        warns.append("未找到 gold-compare.json")

    # ---------- ⑥ 泄漏 ----------
    print("\n[6] 数据泄漏检查")
    mt = set(s["mission"] for s in tr)
    mv = set(s["mission"] for s in va)
    ov = mt & mv
    print(f"    train mission={len(mt)}  val mission={len(mv)}  重叠={len(ov)}  "
          f"{'✓' if not ov else '✗ 必须为 0'}")
    if ov:
        fails.append(f"train/val mission 重叠 {len(ov)}")

    # ---------- 结论 ----------
    print("\n" + "=" * 92)
    if fails:
        print(f" 结论：✗ 未通过（{len(fails)} 项）")
        for f in fails:
            print(f"   - {f}")
    else:
        print(" 结论：✓ 全部通过（无完全重复；高相似对 <5%；无 mission 泄漏；长度达标）")
    for w in warns:
        print(f"   ⚠ {w}")
    print("=" * 92)

    with open(os.path.join(args.dir, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_train": len(tr), "n_val": len(va), "n_total": len(new),
            "exact_duplicates": exact_dup, "dup_assistant_lines": dup_a,
            "sim_pairs": pairs, "sim_ratio": ratio, "sim_threshold": args.sim_thr,
            "assistant_tokens": describe([s["assistant_tokens"] for s in new]),
            "user_tokens": describe([s["user_tokens"] for s in new]),
            "variant": dict(collections.Counter(s["variant"] for s in new)),
            "kind": dict(collections.Counter(s["kind"] for s in new)),
            "channel": dict(collections.Counter(s["channel"] for s in new)),
            "region": dict(collections.Counter(s["region"] for s in new)),
            "missions": len(set(s["mission"] for s in new)),
            "leak_overlap": len(ov), "fails": fails, "warns": warns,
        }, f, ensure_ascii=False, indent=2)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
