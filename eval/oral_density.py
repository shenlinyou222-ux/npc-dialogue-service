# -*- coding: utf-8 -*-
"""对比一代/二代训练数据的口癖密度。"""
import json, io, os, collections

ORAL = ["嘿嘿", "本姑娘", "咱", "呀", "啦", "哇", "呗", "嘛", "哼", "诶", "哦", "咯", "嘿嘿嘿"]

FILES = [
    ("V1 教师蒸馏 100 条", r"E:\habitica-setup\job\distill\train.jsonl"),
    ("V1-v3 教师蒸馏", r"E:\habitica-setup\job\distill\train_v3.jsonl"),
    ("V2 语料重建 1289 条", r"C:\Users\user\Desktop\ds harness\march7_dataset_v2\train.jsonl"),
    ("V2 验证 68 条", r"C:\Users\user\Desktop\ds harness\march7_dataset_v2\val.jsonl"),
]

def assistants(path):
    out = []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            msgs = d.get("messages") or []
            txt = "".join(m.get("content", "") for m in msgs if m.get("role") == "assistant")
            if txt:
                out.append(txt)
    return out

print("%-22s %6s %8s %8s %7s %7s" % ("数据集", "条数", "总字数", "均字数", "感叹号", "每千字口癖"))
print("-" * 72)

rows = []
for name, path in FILES:
    if not os.path.exists(path):
        print("%-22s  (文件不存在)" % name)
        continue
    a = assistants(path)
    if not a:
        print("%-22s  (无 assistant 内容)" % name)
        continue
    chars = sum(len(t) for t in a)
    bang = sum(t.count("！") + t.count("!") for t in a)
    c = collections.Counter()
    for t in a:
        for k in ORAL:
            c[k] += t.count(k)
    total = sum(c.values())
    per1k = total / chars * 1000 if chars else 0
    # 含口癖的样本比例
    hit_samples = sum(1 for t in a if any(k in t for k in ORAL))
    rows.append((name, len(a), chars, c, total, per1k, hit_samples))
    print("%-22s %6d %8d %8.1f %7.2f %8.2f" % (name, len(a), chars, chars / len(a),
                                                bang / len(a), per1k))

print()
print("=== 逐口癖命中（次数 / 每千字）===")
hdr = "%-10s" % "口癖"
for r in rows:
    hdr += " | %-18s" % r[0][:10]
print(hdr)
for k in ORAL:
    line = "%-10s" % k
    any_hit = False
    for r in rows:
        n = r[3].get(k, 0)
        if n:
            any_hit = True
        line += " | %6d (%5.2f)   " % (n, n / r[2] * 1000 if r[2] else 0)
    if any_hit:
        print(line)

print()
print("=== 自称用法对比 ===")
for r in rows:
    n, chars, c = r[1], r[2], r[3]
    a = c.get("本姑娘", 0)
    b = c.get("咱", 0)
    tot = a + b
    print("%-22s 本姑娘=%4d  咱=%4d  占比 本姑娘 %.0f%% / 咱 %.0f%%" %
          (r[0], a, b, a / tot * 100 if tot else 0, b / tot * 100 if tot else 0))

print()
print("=== 含口癖样本占比 ===")
for r in rows:
    print("%-22s %d/%d = %.1f%%" % (r[0], r[6], r[1], r[6] / r[1] * 100))
