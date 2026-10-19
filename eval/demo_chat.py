#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三月七 NPC 对话模型 · 可演示 CLI

关键：本脚本里的 SYSTEM / USER_TMPL 是**逐字对齐训练数据**
（E:\\habitica-setup\\job\\distill\\train.jsonl，即 lora_distill_08b → Qwen3.5-0.8B-March7 用的格式）。
改这里的任何一句，输出形态都会变（会开始出现旁白和括号）。

用法：
  /home/user/vllm-venv/bin/python demo_chat.py                      # 交互模式
  /home/user/vllm-venv/bin/python demo_chat.py -q "你今天怎么这么高兴"
  /home/user/vllm-venv/bin/python demo_chat.py -m /home/user/npc-models/Qwen3.5-0.8B-March7-v3
  /home/user/vllm-venv/bin/python demo_chat.py --n 3 -q "..."        # 同一条问 3 次看稳定性
"""
import argparse, os, sys, time, json

SYSTEM = """【输出规则】
1. 只输出三月七说的话，不要旁白、括号、动作描写。
2. 不要复述对方台词。
3. 长度 10-40 字，最多 2 句。

你是《崩坏：星穹铁道》中的角色「三月七」，星穹列车的成员。
说话方式：自称「本姑娘」「咱」；常用「嘿嘿」「呀」「啦」「哇」；短句、口语、直接表达情绪。"""

USER_TMPL = """【本次场景】{scene}

【相关情报】
{lore}

【对话】
开拓者：{line}

【格式提醒】只输出三月七的台词原句。不要括号、不要旁白、不要动作描写。10-40 字，最多 2 句。"""


def build_prompt(tok, line, scene, lore):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(scene=scene, lore=lore, line=line)},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load(model_path, adapter=None):
    """注意：必须用 AutoModelForImageTextToText 这条入口。

    Qwen3.5 是混合架构，两个入口给出不同的类树：
      AutoModelForCausalLM        -> Qwen3_5ForCausalLM（无 .language_model）
      AutoModelForImageTextToText -> Qwen3_5ForConditionalGeneration（有 .language_model）
    训练脚本走的是后者，LoRA 的 key 也长在后者那棵树里。
    用错入口 → 所有 adapter key 静默失配 → 你其实在跑基座模型。
    """
    import torch
    from transformers import AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText as AutoModel
        entry = "AutoModelForImageTextToText"
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModel
        entry = "AutoModelForCausalLM"

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True,
    )
    print("[load] entry=%s  class=%s" % (entry, type(model).__name__))

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        nz = tot = 0
        for n, p in model.named_parameters():
            if "lora_B" in n and p.numel() > 0:
                tot += 1
                if p.abs().sum().item() != 0:
                    nz += 1
        print("[load] lora_B 模块: %d 个，非零 %d 个" % (tot, nz))
        if tot == 0 or nz == 0:
            raise RuntimeError("adapter 没加载上（lora_B 全零）。加载入口与训练不一致。")
        model = model.merge_and_unload()
        print("[load] adapter merged from %s" % adapter)
    model.eval()
    if adapter is None:
        print("[load] ⚠️ 未指定 --adapter，当前跑的是【基座模型】，不是微调模型。")
        print("[load]    微调请加：--adapter /home/user/npc-models/lora-m7mix-3ep/checkpoint-ep1")
    return tok, model


def gen(tok, model, line, scene="仙舟「罗浮」的甲板上", lore="（无）",
        max_new_tokens=64, temperature=0.7, top_p=0.9, rep=1.1):
    import torch
    prompt = build_prompt(tok, line, scene, lore)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=rep,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    dt = (time.time() - t0) * 1000
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="/home/user/npc-models/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default=None, help="PEFT adapter 目录（可选）")
    ap.add_argument("-q", "--question", default=None, help="单条提问；省略则进交互模式")
    ap.add_argument("--scene", default="仙舟「罗浮」的甲板上")
    ap.add_argument("--lore", default="（无）")
    ap.add_argument("--n", type=int, default=1, help="同一问题重复采样次数")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        sys.exit("模型目录不存在: %s" % args.model)

    print("[load] %s" % args.model)
    t0 = time.time()
    tok, model = load(args.model, args.adapter)
    print("[load] %.1f s   device=%s" % (time.time() - t0, model.device))

    if args.question:
        for i in range(args.n):
            text, dt = gen(tok, model, args.question, args.scene, args.lore,
                           args.max_new_tokens, args.temperature, args.top_p)
            print("\n--- #%d  (%.0f ms) ---" % (i + 1, dt))
            print(text)
        return

    print("\n输入内容回车提问，输入 /exit 退出。\n")
    while True:
        try:
            q = input("开拓者> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q in ("/exit", "/quit"):
            break
        text, dt = gen(tok, model, q, args.scene, args.lore,
                       args.max_new_tokens, args.temperature, args.top_p)
        print("三月七> %s" % text)
        print("        (%.0f ms)\n" % dt)


if __name__ == "__main__":
    main()
