#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
口癖展示 + 格式体检（可演示的核心脚本）

做三件事：
  1. 用 13 条固定场景跑模型，输出对照表（输入 → 三月七说了什么）
  2. 统计口癖命中：嘿嘿 / 本姑娘 / 呀 / 啦 / 哇 / 咱 / 呗 / 嘛 / 哼 / 诶
  3. 体检格式违规：旁白、括号、复述对方台词、超长 —— 算出「格式通过率」

用法：
  /home/user/vllm-venv/bin/python demo_oral.py
  /home/user/vllm-venv/bin/python demo_oral.py -m /home/user/npc-models/Qwen3.5-0.8B-March7-v3
  /home/user/vllm-venv/bin/python demo_oral.py --out 口癖报告.md
"""
import argparse, os, sys, json, re, time, statistics

from demo_chat import load, gen

# 场景：id / 名称 / 玩家台词 / 场景 / 情报 / 期望观察点
SCENES = [
    ("S01", "打招呼", "哟，你今天来得挺早啊。", "星穹列车·观景车厢", "（无）", "开场语气"),
    ("S02", "被夸", "你刚才那个操作挺厉害的。", "仙舟「罗浮」的甲板上", "（无）", "得意 + 自称"),
    ("S03", "惊讶", "前面那个发光的東西是什么？", "雅利洛-VI·雪原", "【星核】被冰封七百年的灾厄源头", "惊讶感叹"),
    ("S04", "拒绝", "能帮我搬一下这个箱子吗？", "黑塔空间站·收容舱段", "（无）", "推脱但可爱"),
    ("S05", "撒娇", "我有点累了，能休息一下吗？", "星穹列车·客房车厢", "（无）", "软化 + 语气词"),
    ("S06", "生气", "你刚才把我那份蛋糕吃了。", "星穹列车·餐车", "（无）", "假生气，不能真凶"),
    ("S07", "战斗", "敌人从右边绕过来了。", "贝洛伯格·残响回廊", "【裂界造物】受星核侵蚀的怪物", "短促、催人"),
    ("S08", "告别", "我先走了，明天见。", "星穹列车·站台", "（无）", "收尾语气"),
    ("S09", "不确定", "你知道列车下一站去哪吗？", "星穹列车·观景车厢", "（无）", "**不该编造**，应承认不知道"),
    ("S10", "自我介绍", "还没问过你叫什么名字。", "星穹列车·观景车厢", "（无）", "承担身份设定"),
    ("S11", "安慰", "我这次考试又没过。", "仙舟「罗浮」·长乐天", "（无）", "共情，不是套话"),
    ("S12", "极短回应", "嗯。", "星穹列车·走廊", "（无）", "短输入能不能不崩"),
    ("S13", "诱导幻觉", "你还记得我们上次一起买的那把剑吗？", "星穹列车·观景车厢", "（无）", "**已知短板**：会不会编"),
]

ORAL = ["嘿嘿", "本姑娘", "咱", "呀", "啦", "哇", "呗", "嘛", "哼", "诶", "哦", "咯"]
BAD_BRACKET = ["（", "）", "(", ")", "【", "】", "*"]
NARRATION_HINT = ["旁白", "动作", "心想", "暗道"]


def check(text):
    """返回 (是否格式通过, 问题列表)"""
    probs = []
    for b in BAD_BRACKET:
        if b in text:
            probs.append("含括号/符号 %s" % b)
    for n in NARRATION_HINT:
        if n in text:
            probs.append("含旁白词")
            break
    if text.startswith("三月七") or "：" in text or ":" in text:
        probs.append("疑似带说话人前缀")
    if len(text) > 60:
        probs.append("超长 %d 字" % len(text))
    if len(text.strip()) == 0:
        probs.append("空输出")
    return (len(probs) == 0), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="/home/user/npc-models/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default=None, help="PEFT adapter 目录")
    ap.add_argument("--out", default=None, help="输出 markdown 路径")
    ap.add_argument("-n", type=int, default=1, help="每条场景采样次数")
    args = ap.parse_args()

    print("[load] %s" % args.model)
    t0 = time.time()
    tok, model = load(args.model, args.adapter)
    print("[load] %.1f s\n" % (time.time() - t0))

    rows = []
    lat = []
    hits = {k: 0 for k in ORAL}
    passed = 0
    for sid, name, line, scene, lore, note in SCENES:
        outs = []
        for _ in range(args.n):
            text, dt = gen(tok, model, line, scene, lore)
            out = text.strip()
            outs.append(out)
            lat.append(dt)
            for k in ORAL:
                if k in out:
                    hits[k] += 1
            ok, probs = check(out)
            if ok:
                passed += 1
            rows.append({
                "id": sid, "name": name, "in": line, "out": out,
                "chars": len(out), "ms": int(dt), "ok": ok,
                "probs": probs, "note": note,
                "hit": [k for k in ORAL if k in out],
            })
            print("%s %-6s %3dms %s | %s%s" % (
                sid, name, dt, "OK " if ok else "BAD", out,
                "" if ok else "   <<< " + "; ".join(probs)))

    total = len(rows)
    rate = passed / total * 100 if total else 0
    p50 = statistics.median(lat) if lat else 0
    cjk_lens = [r["chars"] for r in rows]
    avg_len = sum(cjk_lens) / len(cjk_lens) if cjk_lens else 0

    print("\n================ 汇总 ================")
    print("样本数        : %d" % total)
    print("格式通过率    : %.1f%%  (%d/%d)" % (rate, passed, total))
    print("平均字数      : %.1f" % avg_len)
    print("延迟 p50      : %.0f ms" % p50)
    print("口癖命中      : " + "  ".join("%s=%d" % (k, v) for k, v in hits.items() if v))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("# 三月七 LoRA · 口癖与格式展示\n\n")
            fh.write("模型：`%s`　采样：%d 条　温度 0.7 / top-p 0.9\n\n" % (os.path.basename(args.model), total))
            fh.write("| 指标 | 值 |\n|---|---|\n")
            fh.write("| 格式通过率 | **%.1f%%**（%d/%d） |\n" % (rate, passed, total))
            fh.write("| 平均字数 | %.1f |\n" % avg_len)
            fh.write("| 延迟 p50 | %.0f ms |\n\n" % p50)
            fh.write("## 口癖命中分布\n\n| 口癖 | 命中 |\n|---|---:|\n")
            for k, v in sorted(hits.items(), key=lambda x: -x[1]):
                if v:
                    fh.write("| %s | %d |\n" % (k, v))
            fh.write("\n## 逐条对照\n\n| ID | 场景 | 玩家说 | 三月七说 | 字数 | 口癖 | 格式 |\n")
            fh.write("|---|---|---|---|---:|---|---|\n")
            for r in rows:
                fh.write("| %s | %s | %s | %s | %d | %s | %s |\n" % (
                    r["id"], r["name"], r["in"].replace("|", "｜"), r["out"].replace("|", "｜"),
                    r["chars"], "/".join(r["hit"]) or "—",
                    "OK" if r["ok"] else "BAD"))
            fh.write("\n> 生成命令：`/home/user/vllm-venv/bin/python demo_oral.py --out 口癖报告.md`\n")
        print("\n[write] %s" % args.out)


if __name__ == "__main__":
    main()
