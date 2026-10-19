# -*- coding: utf-8 -*-
"""
构建最终训练集：原始语料（三月七）+ LLM 合成（二代/三代），带清洗。

清洗项：
  1. 只留 character == 三月七
  2. assistant 非空、长度 4–150 字
  3. 去掉含旁白标记（（）、【】、*）的 assistant
  4. 去掉带说话人前缀的 assistant（"三月七："）
  5. 跨来源去重（normalize(user)+normalize(assistant)）
  6. 统一 system prompt 为语料线那套
  7. mission 级防泄漏（corpus 用 mission；llm 用 __llm__ 伪 mission，全进 train）
  8. **用 tokenizer 实测前缀匹配率，必须 100%**

输出：/tmp/m7mix/{train.jsonl,val.jsonl} + report.json
"""
import json, io, os, re, sys, random, collections

CORPUS_TRAIN = "/mnt/c/Users/user/Desktop/ds harness/lora_hsr/data/train.jsonl"
CORPUS_VAL = "/mnt/c/Users/user/Desktop/ds harness/lora_hsr/data/val.jsonl"
DIST = "/mnt/e/habitica-setup/job/distill"
TRAIN_V1 = os.path.join(DIST, "train.jsonl")          # 一代配对（有 messages）
TEACHER_V1 = os.path.join(DIST, "teacher_outputs.json")
TEACHER_V2 = os.path.join(DIST, "teacher_v2.json")
TRAIN_V3 = os.path.join(DIST, "train_v3.jsonl")       # 三代配对
OUT = "/tmp/m7mix"
MODEL = "/home/user/npc-models/Qwen3.5-0.8B"

SYS = ("你是《崩坏：星穹铁道》中的角色「三月七」。"
       "请严格保持该角色的说话风格、用词习惯与身份设定，用中文回复。")

ORAL = ["嘿嘿", "本姑娘", "咱", "呀", "啦", "哇", "呗", "嘛", "哼", "诶", "哦", "咯"]
BAD = ["（", "）", "(", ")", "【", "】", "*", "旁白"]

random.seed(20251019)


def load_jsonl(p):
    out = []
    for line in io.open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def norm(s):
    return re.sub(r"\s+", "", s)


def clean_assistant(a):
    """返回 (是否通过, 原因)"""
    if not a or not a.strip():
        return False, "空"
    a = a.strip()
    for b in BAD:
        if b in a:
            return False, "含旁白/括号:" + b
    if re.match(r"^(三月七|开拓者|丹恒|姬子|瓦尔特|帕姆)\s*[:：]", a):
        return False, "带说话人前缀"
    n = len(a)
    if n < 4:
        return False, "过短"
    if n > 150:
        return False, "过长"
    return True, ""


def to_sample(user, assistant, mission, src):
    return {
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "mission": mission,
        "source": src,
        "character": "三月七",
    }


stat = collections.Counter()
seen = set()


def add(pool, user, assistant, mission, src):
    ok, why = clean_assistant(assistant)
    if not ok:
        stat["drop_" + why.split(":")[0]] += 1
        return
    key = norm(user)[:400] + "|" + norm(assistant)
    if key in seen:
        stat["drop_重复"] += 1
        return
    seen.add(key)
    pool.append(to_sample(user, assistant, mission, src))
    stat["keep_" + src] += 1


# ---------- 1. 语料线 ----------
train, val = [], []
for rec in load_jsonl(CORPUS_TRAIN):
    if rec.get("character") != "三月七":
        stat["skip_非三月七"] += 1
        continue
    msgs = rec.get("messages") or []
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    add(train, u, a, rec.get("mission", "__none__"), "corpus")

for rec in load_jsonl(CORPUS_VAL):
    if rec.get("character") != "三月七":
        continue
    msgs = rec.get("messages") or []
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    ok, _ = clean_assistant(a)
    if ok:
        val.append(to_sample(u, a, rec.get("mission", "__none__"), "corpus"))

corpus_train_used = len(train)
corpus_val_used = len(val)

# ---------- 2. 重建二代配对（用一代的 prompt + 二代的输出） ----------
v1 = json.load(io.open(TEACHER_V1, encoding="utf-8"))
v2 = json.load(io.open(TEACHER_V2, encoding="utf-8"))
paired = load_jsonl(TRAIN_V1)

# 校验顺序是否对齐：train.jsonl[i] 的 assistant == teacher_outputs[i].out
aligned = 0
for i, rec in enumerate(paired):
    a = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
    if i < len(v1) and norm(a) == norm(v1[i].get("out", "")):
        aligned += 1
stat["v1配对对齐"] = aligned
stat["v1配对数"] = len(paired)

v2_added = 0
if aligned == len(paired):
    for i, rec in enumerate(paired):
        pid = v1[i].get("id")
        out2 = v2.get(pid)
        if not isinstance(out2, str):
            continue
        u = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        add(train, u, out2, "__llm_v2__", "llm_v2")
        v2_added += 1

# ---------- 3. 三代配对 ----------
v3_added = 0
for rec in load_jsonl(TRAIN_V3):
    msgs = rec.get("messages") or []
    u = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    add(train, u, a, "__llm_v3__", "llm_v3")
    v3_added += 1

# ---------- 4. 统计 ----------
def density(rs):
    ch = sum(len(next(m["content"] for m in r["messages"] if m["role"] == "assistant")) for r in rs)
    orl = sum(next(m["content"] for m in r["messages"] if m["role"] == "assistant").count(k)
              for r in rs for k in ORAL)
    return ch, orl, (orl / ch * 1000 if ch else 0)


tc, to, td = density(train)
vc, vo, vd = density(val)

# ---------- 5. 前缀匹配实测 ----------
prefix_ok = prefix_bad = 0
prefix_err = None
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    for r in train[:800]:
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        fid = tok(full, add_special_tokens=False)["input_ids"]
        if fid[:len(pid)] == pid:
            prefix_ok += 1
        else:
            prefix_bad += 1
except Exception as e:
    prefix_err = repr(e)[:300]

# ---------- 6. 写盘 ----------
os.makedirs(OUT, exist_ok=True)
random.shuffle(train)
with io.open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as fh:
    for r in train:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with io.open(os.path.join(OUT, "val.jsonl"), "w", encoding="utf-8") as fh:
    for r in val:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# 泄漏检查
mt = set(r["mission"] for r in train)
mv = set(r["mission"] for r in val)
leak = len(mt & mv)

rep = {
    "train_n": len(train),
    "val_n": len(val),
    "train_chars": tc,
    "train_oral_per1k": round(td, 2),
    "val_oral_per1k": round(vd, 2),
    "来源": dict(stat),
    "mission_overlap": leak,
    "prefix_ok": prefix_ok,
    "prefix_bad": prefix_bad,
    "prefix_err": prefix_err,
}
json.dump(rep, io.open(os.path.join(OUT, "report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("=" * 66)
print("清洗报告")
print("=" * 66)
print("train %d 条 / %d 字     val %d 条" % (len(train), tc, len(val)))
print("train 口癖 %.2f /千字    val 口癖 %.2f /千字   （真值 ≈15.2）" % (td, vd))
print("mission 重叠（必须 0）: %d" % leak)
print("前缀匹配: ok=%d bad=%d %s" % (prefix_ok, prefix_bad, prefix_err or ""))
print("-" * 66)
for k, v in sorted(stat.items()):
    print("  %-22s %d" % (k, v))
print("-" * 66)
print("输出:", OUT)
