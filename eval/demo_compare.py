# -*- coding: utf-8 -*-
"""
基座 vs 微调（ep1）对照演示。

两个要点：
1. **system prompt 必须用训练时那套**（语料线：你是…三月七…请严格保持…），
   不是老的三分段【输出规则】。用错 prompt，模型看到的是分布外的输入。
2. **必须带场景上下文**。这个模型是在「【任务：…】【场景：…】+ 多行对话」的格式上训的，
   单句提问是分布外的，会拉低表现。

用法：
  /home/user/vllm-venv/bin/python demo_compare.py
  /home/user/vllm-venv/bin/python demo_compare.py --adapter .../checkpoint-ep1
"""
import argparse, os, sys, time, torch
from transformers import AutoTokenizer

SYS = ("你是《崩坏：星穹铁道》中的角色「三月七」。"
       "请严格保持该角色的说话风格、用词习惯与身份设定，用中文回复。")

# 场景按训练数据格式写：任务 + 场景 + 若干行对话
CASES = [
    ("列车·回礼",
     "【任务：安灵布奠，天清路远】【场景：挑选人们的回礼】\n"
     "三月七：这是梦茗小姐为你准备的茶，以你的名字命名。\n"
     "开拓者：那杯茶呢？\n"
     "帕姆：咱已经把那杯茶给喝了呀。\n"
     "开拓者：那我喝什么？"),
    ("罗浮·找向导",
     "【任务：腐烂或燃烧】【场景：寻找「地火」安排的向导】\n"
     "希儿：…虎克，是你？\n"
     "虎克：嘿嘿，是我呀！\n"
     "开拓者：你认识她？"),
    ("雪原·遇敌",
     "【任务：永冬孤城】【场景：贝洛伯格城外的雪原】\n"
     "丹恒：前面的雪里有东西在动。\n"
     "开拓者：数量不少，怎么办？"),
    ("被质疑",
     "【任务：日常闲聊】【场景：星穹列车观景车厢】\n"
     "开拓者：其实你根本不记得我叫什么吧？\n"
     "三月七：诶？\n"
     "开拓者：说吧。"),
    ("危险判断",
     "【任务：星核之影】【场景：雅利洛-VI 的废墟】\n"
     "开拓者：那团紫光是什么？\n"
     "丹恒：别靠近，那是裂界的入口。"),
]


def build(tok, user):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def gen(tok, model, user, max_new=72, temp=0.7, top_p=0.9, rep=1.1):
    ids = tok(build(tok, user), return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=True, temperature=temp, top_p=top_p,
                             repetition_penalty=rep,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip(), \
        (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="/home/user/npc-models/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default="/home/user/npc-models/lora-m7mix-3ep/checkpoint-ep1")
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText as AutoModel
    print("[load] base =", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16,
                                      device_map="auto", low_cpu_mem_usage=True)
    print("[load] class =", type(model).__name__)

    # 先跑基座
    base_out = []
    for name, ctx in CASES:
        txt, ms = gen(tok, model, ctx, temp=args.temp)
        base_out.append(txt)
        print("  [base] %-10s %s" % (name, txt))

    # 再挂 adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    nz = sum(1 for n, p in model.named_parameters()
             if "lora_B" in n and p.abs().sum().item() != 0)
    print("[load] adapter=%s  lora_B 非零=%d" % (os.path.basename(args.adapter), nz))
    if nz == 0:
        sys.exit("adapter 没加载上")
    model = model.merge_and_unload()
    model.eval()

    tuned_out = []
    for name, ctx in CASES:
        txt, ms = gen(tok, model, ctx, temp=args.temp)
        tuned_out.append((txt, ms))
        print("  [tune] %-10s %s" % (name, txt))

    print("\n" + "=" * 78)
    print("对照（温度 %.2f / top-p 0.9 / rep 1.1 / system = 语料线）" % args.temp)
    print("=" * 78)
    for (name, ctx), b, (t, ms) in zip(CASES, base_out, tuned_out):
        last = ctx.strip().split("\n")[-1]
        print("\n【%s】" % name)
        print("  上文末句 : %s" % last)
        print("  基座     : %s" % b)
        print("  微调 ep1 : %s   (%.0f ms)" % (t, ms))


if __name__ == "__main__":
    main()
