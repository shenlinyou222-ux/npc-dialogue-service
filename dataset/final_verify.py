# -*- coding: utf-8 -*-
"""交付前最终校验：检查报告中的关键数字是否与实际数据一致。"""
import json, os, collections, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

tr, va = jl(os.path.join(HERE, "train.jsonl")), jl(os.path.join(HERE, "val.jsonl"))
new = tr + va
ok = True

def chk(label, got, want):
    global ok
    good = got == want
    ok = ok and good
    print(f"  {'✓' if good else '✗'} {label:46s} got={got}  want={want}")

print("=== 交付前最终校验 ===")
chk("train 条数", len(tr), 1289)
chk("val 条数", len(va), 68)
chk("合计条数", len(new), 1357)
chk("覆盖 mission 数", len(set(s["mission"] for s in new)), 207)
chk("train+val mission 重叠", len(set(s["mission"] for s in tr) & set(s["mission"] for s in va)), 0)

# 长度
chk("assistant token 均值(约)", round(statistics.mean(s["assistant_tokens"] for s in new)), 29)
a = sorted(s["assistant_tokens"] for s in new)
chk("assistant token p50", a[len(a)//2], 27)

# variant
v = collections.Counter(s["variant"] for s in new)
chk("scene_dialogue", v["scene_dialogue"], 768)
chk("bare_line", v["bare_line"], 234)
chk("bare_action", v["bare_action"], 215)
chk("scene_action", v["scene_action"], 140)

# 元数据污染
chk("含【任务：】的样本", sum(1 for s in new if "【任务：" in s["messages"][1]["content"]), 0)
chk("含 ▇ 的样本", sum(1 for s in new if "▇" in s["messages"][1]["content"]), 0)
chk("含 ▇ 的 section 字段", sum(1 for s in new if "▇" in (s.get("section") or "")), 0)
chk("合成上下文样本数", sum(s["synthetic_context"] for s in new), 85)

# 完全重复
c = collections.Counter((s["messages"][1]["content"], s["messages"][2]["content"]) for s in new)
chk("完全重复样本", sum(x-1 for x in c.values() if x > 1), 0)

# 结构自洽：结构化样本（scene_*）里不能有裸行；
# bare_line / bare_action 形态本来就是"无前缀的裸输入"，不应计入。
import re
SPK = re.compile(r"^([^：\n]{1,12})：")
ACT = re.compile(r"^（.+）$")
bad = 0
for s in new:
    if not s["variant"].startswith("scene_"):
        continue
    u = s["messages"][1]["content"].split("\n")
    body = u[1:] if u and u[0].startswith("【场景：") else u
    for ln in body:
        ln = ln.strip()
        if not ln:
            continue
        if not (SPK.match(ln) or ACT.match(ln)):
            bad += 1
chk("结构化(scene_*)样本里的裸行", bad, 0)
chk("结构化样本总数", sum(1 for s in new if s["variant"].startswith("scene_")), 768 + 140)

# 三行 message 结构 + assistant 非空
chk("结构异常样本", sum(1 for s in new if len(s["messages"]) != 3 or not s["messages"][2]["content"].strip()), 0)
chk("system 段一致", len(set(s["messages"][0]["content"] for s in new)), 1)

print()
print("全部通过 ✓" if ok else "存在不一致 ✗")
sys.exit(0 if ok else 1)
