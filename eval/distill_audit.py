# -*- coding: utf-8 -*-
"""把 distill 目录下所有数据集逐个验明：是语料抽取还是大模型生成？口癖多少？"""
import json, io, os, re, glob

DIST = r"E:\habitica-setup\job\distill"
CORPUS = [
    r"C:\Users\user\Desktop\ds harness\game_dialogue_corpus_wiki\by_character\三月七.jsonl",
    r"C:\Users\user\Desktop\ds harness\game_dialogue_corpus_wiki\dialogue_story.jsonl",
    r"C:\Users\user\Desktop\ds harness\game_dialogue_corpus_wiki\dialogue_full.jsonl",
    r"C:\Users\user\Desktop\ds harness\game_dialogue_corpus_wiki\lora_samples.jsonl",
]
ORAL = ["嘿嘿", "本姑娘", "咱", "呀", "啦", "哇", "呗", "嘛", "哼", "诶", "哦", "咯"]

def norm(s):
    return re.sub(r"\s+", "", s.replace("\u3000", " "))

# 语料 blob
chunks = []
for p in CORPUS:
    if not os.path.exists(p):
        continue
    for line in io.open(p, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            chunks.append(line)
            continue
        def walk(o):
            if isinstance(o, str):
                chunks.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
blob = norm("".join(chunks))
print("语料 blob: %d 字符\n" % len(blob))

def load_any(p):
    """返回 assistant 文本列表"""
    out = []
    if p.endswith(".jsonl"):
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict) and "messages" in d:
                out.extend(m["content"] for m in d["messages"] if m.get("role") == "assistant")
            elif isinstance(d, dict) and "out" in d:
                out.append(d["out"])
    else:
        d = json.load(io.open(p, encoding="utf-8"))
        if isinstance(d, list):
            for x in d:
                if isinstance(x, dict):
                    if "out" in x:
                        out.append(x["out"])
                    elif "messages" in x:
                        out.extend(m["content"] for m in x["messages"] if m.get("role") == "assistant")
                    elif "content" in x:
                        out.append(x["content"])
        elif isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str):
                    out.append(v)
    return out

files = sorted(glob.glob(os.path.join(DIST, "*.jsonl")) + glob.glob(os.path.join(DIST, "*.json")))
print("%-30s %6s %8s %8s %10s %s" % ("文件", "条数", "总字数", "每千字口癖", "语料命中率", "判定"))
print("-" * 92)
for p in files:
    try:
        a = load_any(p)
    except Exception as e:
        print("%-30s 读取失败 %s" % (os.path.basename(p), e))
        continue
    if not a:
        print("%-30s (无 assistant 文本)" % os.path.basename(p))
        continue
    chars = sum(len(t) for t in a)
    c = sum(t.count(k) for t in a for k in ORAL)
    per1k = c / chars * 1000 if chars else 0
    # 逐行语料命中
    total = hit = 0
    for b in a:
        for ln in b.split("\n"):
            k = norm(ln)
            k = re.sub(r"^[^：]{1,12}：", "", k)
            if len(k) < 4:
                continue
            total += 1
            if k in blob:
                hit += 1
    rate = hit / total * 100 if total else 0
    verdict = "语料抽取" if rate >= 90 else ("大模型生成" if rate < 60 else "混合/待判")
    print("%-30s %6d %8d %10.2f %9.1f%% %s" % (
        os.path.basename(p), len(a), chars, per1k, rate, verdict))
