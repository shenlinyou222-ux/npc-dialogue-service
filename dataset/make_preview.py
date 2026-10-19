# -*- coding: utf-8 -*-
"""
make_preview.py —— 生成 20 条样本的可读预览（samples_preview.md）+ 机器可读 JSON，
供 report.md 的人工评估章节使用。

抽样方式：对 train+val 合并池做固定种子的分层随机抽样（按 variant 分层），
保证 4 种输入形态都被覆盖。
"""
import argparse
import collections
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N = 20


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HERE)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    for name, tag in (("train.jsonl", "train"), ("val.jsonl", "val")):
        for r in iter_jsonl(os.path.join(args.dir, name)):
            r["_split"] = tag
            rows.append(r)

    # 按 variant 分层抽样，保证 4 种形态都出现
    by_v = collections.defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)
    rng = random.Random(args.seed)
    for v in by_v:
        by_v[v].sort(key=lambda r: r["messages"][2]["content"])
        rng.shuffle(by_v[v])

    picked, i = [], 0
    vs = ["scene_dialogue", "bare_action", "bare_line", "scene_action"]
    while len(picked) < args.n and any(by_v[v] for v in vs):
        v = vs[i % len(vs)]
        if by_v[v]:
            picked.append(by_v[v].pop())
        i += 1
    rng.shuffle(picked)

    lines = ["# 三月七 LoRA 训练数据 v2 —— 20 条抽样预览", "",
             f"来源：`train.jsonl` + `val.jsonl`（分层随机抽样，seed={args.seed}）",
             "抽样刻意覆盖 4 种输入形态（对齐推理实测的 10 个场景）。", "",
             "**格式说明**", "",
             "- `【场景：X】` = 从 `region` + `section` + 上下文关键词推断的场景描述，**不含任何 Wiki 任务名**",
             "- `说话人：台词` = 保留说话人的对话历史（玩家说话人统一为「开拓者」）",
             "- `（…）` = 动作/旁白，对应推理时的 `（开拓者朝你走近了两步）`",
             "- `bare_*` 形态 = 推理实测中「没有【场景：】前缀」的裸输入", "", "---", ""]

    jout = []
    for k, r in enumerate(picked, 1):
        m = {x["role"]: x["content"] for x in r["messages"]}
        lines.append(f"## 样本 {k}　`{r['variant']}`　({r['_split']})")
        lines.append("")
        lines.append(f"- mission：`{r['mission']}`　region：`{r['region'] or '—'}`　"
                     f"channel：`{r['channel']}`　kind：`{r['kind']}`")
        lines.append(f"- 推断场景：`{r['scene']}`　上下文轮数：{r['ctx_turns']}　"
                     f"assistant 行数：{r['assistant_lines']}　"
                     f"token：user={r['user_tokens']} / assistant={r['assistant_tokens']}"
                     + ("　**synthetic_context**" if r.get("synthetic_context") else ""))
        lines.append("")
        lines.append("```text")
        lines.append("[system]")
        lines.append(m["system"])
        lines.append("")
        lines.append("[user]")
        lines.append(m["user"])
        lines.append("")
        lines.append("[assistant]")
        lines.append(m["assistant"])
        lines.append("```")
        lines.append("")
        jout.append({
            "idx": k, "variant": r["variant"], "split": r["_split"],
            "mission": r["mission"], "region": r["region"], "channel": r["channel"],
            "kind": r["kind"], "scene": r["scene"], "ctx_turns": r["ctx_turns"],
            "assistant_lines": r["assistant_lines"],
            "user_tokens": r["user_tokens"], "assistant_tokens": r["assistant_tokens"],
            "synthetic_context": bool(r.get("synthetic_context")),
            "system": m["system"], "user": m["user"], "assistant": m["assistant"],
        })

    with open(os.path.join(args.dir, "samples_preview.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.dir, "samples_preview.json"), "w", encoding="utf-8") as f:
        json.dump(jout, f, ensure_ascii=False, indent=2)
    print(f"写出 samples_preview.md / samples_preview.json（{len(picked)} 条）")
    print("形态分布:", collections.Counter(x["variant"] for x in jout))


if __name__ == "__main__":
    main()
