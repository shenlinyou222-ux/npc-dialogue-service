# -*- coding: utf-8 -*-
"""
build_dataset.py —— 为《崩坏：星穹铁道》角色「三月七」构建 LoRA 训练数据 v2。

逐条对应父代理诊断出的 4 个问题：

① 格式与推理一致
   user 段 = 【场景：X】 + 带说话人的前 2-4 轮对话；绝不出现【任务：XXX】。
   并且**同时覆盖推理实测的 4 种输入形态**（见 variant 字段）：
     scene_dialogue : 【场景：X】\n说话人：台词…        （推理场景 ④⑤⑦⑧⑨）
     scene_action   : 【场景：X】\n（动作）              （推理场景 ⑥）
     bare_action    : （动作）                          （推理场景 ①）
     bare_line      : 一句话，无任何前缀                 （推理场景 ②③⑩）

② 合并冗余
   最小样本单位 = 「三月七发言游程的连续片段」，而不是"一行一条"。
   一个 mission 产出 2-5 条，而不是 29 条。

③ 输出不过短
   连续多轮台词合并成一段 assistant（--max-merge-lines，默认 2 轮），
   长游程按 cap 切块并让前一块进入上下文（"同一场景的不同切片"）。
   经 calibrate.py 标定：assistant token 均值 ≈ 27-29，落在 20-30 目标区间。

④ 多样性
   4 种输入形态 × 2/3/4 轮上下文窗口 × 3 种 channel(mission/sms/voice)
   × region 分层配额 × 角色语音层（战斗/日常）+「三月七」别名台词层。

数据源（只读）：
  .../game_dialogue_corpus_wiki/by_mission/*.jsonl   有序对话 + 旁白
  .../game_dialogue_corpus_wiki/dialogue_full.jsonl  含 kind=voice 角色语音

输出：train.jsonl / val.jsonl / build_stats.json
注意：脚本按"在 WSL 里运行"设计，路径用 /mnt/c/... 形式。
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
from difflib import SequenceMatcher

CORPUS = "/mnt/c/Users/user/Desktop/ds harness/game_dialogue_corpus_wiki"
HERE = os.path.dirname(os.path.abspath(__file__))

CHAR = "三月七"
PLAYER = "开拓者"
SYS = "你是《崩坏：星穹铁道》中的角色「三月七」。请严格保持该角色的说话风格、用词习惯与身份设定，用中文回复。"

ALIASES = {"三月七", "「三月七」"}     # 语料里同一角色的两种标注
NARRATION_MAX_CHARS = 50              # 超过此长度的旁白视为散文旁白，作上下文边界
ACTION_MIN_CHARS = 8                  # 动作输入的旁白长度下限
ACTION_MAX_CHARS = 40                 # 动作输入的旁白长度上限

GENERIC_SECTIONS = {"剧情内容", "", None, "任务接取短信", "直播连线"}

# 语料旁白里混入的大量**系统/UI 提示**（不是叙事动作），必须排除，
# 否则会产出「（获得物品）」这种低信息量的输入。
UI_BLOCK_RE = re.compile(
    r"^(进入战斗|结束战斗|进入战斗并获胜|获得胜利|赢得胜利|战斗胜利|获得物品|获得道具|获得|"
    r"提交|提交道具|提交物品|再次对话|选项同上|投掷骰子|击败|胜利|对话|一段时间后|与此同时|"
    r"与.{1,8}对话|我方|对方|敌方|任务完成|任务失败|已开启|已离开|已获得|解锁|"
    r"进入第二阶段|进入第三阶段|调查已|点击|选择)|"
    r"血量|剩余|回合数|气势值|增益|减益|冷却|护盾值|韧性"
)


def log(*a):
    print(*a, flush=True)


# 语料里残存的"性别占位符"——真实游戏会用玩家性别替换，留在训练数据里不自然。
PLACEHOLDER_FIX = [("他/她", "他"), ("她/他", "他"),
                   ("开拓者先生/小姐", "开拓者"), ("开拓者小姐/先生", "开拓者")]
_norm_hits = collections.Counter()


def norm_text(t):
    if not t:
        return t
    for a, b in PLACEHOLDER_FIX:
        if a in t:
            _norm_hits[a] += t.count(a)
            t = t.replace(a, b)
    return t


def md5i(s, mod=None):
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:12], 16)
    return h if mod is None else h % mod


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_tokenizer(model_dir):
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        return lambda s: len(tk.encode(s, add_special_tokens=False)), tk
    except Exception as e:
        log(f"[warn] tokenizer 加载失败({e})，退化为字符估算")
        return lambda s: max(1, int(len(s) * 0.62)), None


# ============================================================================
# 场景推断：region（地点）+ section/关键词（情境）
# ============================================================================
LOC_MAP = [
    ("雅利洛-Ⅵ-磐岩镇", "贝洛伯格·磐岩镇"),
    ("雅利洛-Ⅵ-铁卫禁区", "贝洛伯格·铁卫禁区"),
    ("雅利洛-Ⅵ", "贝洛伯格"),
    ("星穹列车", "星穹列车车厢"),
    ("空间站「黑塔」-主控舱段", "空间站「黑塔」·主控舱段"),
    ("空间站「黑塔」", "空间站「黑塔」"),
    ("仙舟「罗浮」-竞锋舰", "仙舟罗浮·竞锋舰"),
    ("仙舟「罗浮」", "仙舟罗浮"),
    ("匹诺康尼", "匹诺康尼"),
    ("翁法罗斯", "翁法罗斯"),
    ("二相乐园", "二相乐园"),
    ("千星城", "千星城"),
]

MOOD_RULES = [
    (["战斗", "决斗", "迎战", "敌人", "魔物", "怪物", "危险", "小心", "袭击", "开打", "迎击", "擂台", "对决"],
     "气氛紧张"),
    (["告别", "辞别", "再见", "送别", "分别"], "即将分别"),
    (["记忆", "过去", "以前", "忘记", "想起", "失忆"], "夜谈"),
    (["拍照", "照片", "相机", "合影"], "闲聊"),
    (["玩笑", "哈哈", "闹", "别闹", "逗"], "闲聊"),
]


def location_of(region):
    if not region:
        return None
    for pat, name in LOC_MAP:
        if region.startswith(pat) or pat in region:
            return name
    return region.split("、")[0].strip() or None


CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def section_ok(sec):
    """section 是否适合当场景限定语。

    排除泛化值，以及游戏里被涂黑的文本（如「▇ ▇ ▇ ▇ ▇ ▇」）——
    它们当成场景描述会很难看。
    """
    if not sec or sec in GENERIC_SECTIONS:
        return False
    if len(sec) > 16 or "▇" in sec:
        return False
    return len(CJK_RE.findall(sec)) >= 3


def mood_of(text, section, channel):
    if channel == "sms":
        return "手机短信聊天"
    hay = (section or "") + " " + text
    for kws, qual in MOOD_RULES:
        if any(k in hay for k in kws):
            return qual
    sec = (section or "").strip()
    if section_ok(sec):
        return sec
    return None


def build_scene(meta, ctx_text, channel, section):
    loc = location_of(meta.get("region"))
    qual = mood_of(ctx_text, section, channel)
    if loc and qual:
        return f"{loc}，{qual}"
    return loc or qual or "旅途中"


# ============================================================================
# 角色语音（kind=voice）的合成上下文
# 语料里的角色语音是孤立台词、没有上下文。这里按 section 合成"忠实但合成"的
# 触发情境，并刻意对齐推理实测的 10 个场景（战斗前/战斗后/危险/告别/看雪…），
# 使训练与推理在语义分布上也对齐。此类样本 synthetic_context=1，可随时剔除。
# ============================================================================
VOICE_SYNTH = [
    ("初次见面",         "星穹列车车厢，初次见面", "dialogue", "你就是三月七吧？我是新来的。"),
    ("问候",             "星穹列车车厢，日常",     "action",   "开拓者朝你走近了两步"),
    ("道别",             "星穹列车车厢，夜晚",     "dialogue", "我先去休息了，明天见。"),
    ("关于自己•名字",    "星穹列车车厢，夜谈",     "dialogue", "说起来，三月七这个名字是谁取的？"),
    ("关于自己•过去",    "星穹列车车厢，夜谈",     "dialogue", "你还记得以前的事吗？"),
    ("闲谈•果汁",        "星穹列车车厢，闲聊",     "dialogue", "你手边那杯是什么？"),
    ("闲谈•照片",        "星穹列车车厢，闲聊",     "dialogue", "又在翻相册？"),
    ("爱好",             "星穹列车车厢，闲聊",     "dialogue", "你平时都喜欢做什么？"),
    ("烦恼",             "星穹列车车厢，闲聊",     "dialogue", "看你好像有点闷。"),
    ("分享•习剑",        "仙舟罗浮，习剑归来",     "dialogue", "今天学剑学得怎么样？"),
    ("分享",             "星穹列车车厢，夜谈",     "dialogue", "最近有什么想说的吗？"),
    ("见闻",             "星穹列车车厢，闲聊",     "dialogue", "路上有遇到什么有趣的事吗？"),
    ("关于丹恒•二",      "星穹列车车厢，闲聊",     "dialogue", "你现在怎么看丹恒？"),
    ("关于丹恒",         "星穹列车车厢，闲聊",     "dialogue", "丹恒是个怎样的人？"),
    ("关于姬子",         "星穹列车车厢，闲聊",     "dialogue", "你觉得姬子姐怎么样？"),
    ("关于瓦尔特",       "星穹列车车厢，闲聊",     "dialogue", "杨叔以前是做什么的？"),
    ("关于帕姆",         "星穹列车车厢，闲聊",     "dialogue", "列车长怎么样？"),
    ("关于彦卿",         "仙舟罗浮，习剑之后",     "dialogue", "彦卿师父教得严吗？"),
    ("关于云璃",         "仙舟罗浮，习剑之后",     "dialogue", "云璃师父教得怎么样？"),
    ("关于星期日",       "星穹列车车厢，闲聊",     "dialogue", "你怎么看星期日？"),
    ("星魂激活",         "星穹列车，力量提升",     "dialogue", "你好像变强了？"),
    ("角色晋阶",         "星穹列车，力量提升",     "dialogue", "感觉你状态不错。"),
    ("角色满级",         "星穹列车，力量提升",     "dialogue", "你已经成长到极限了吧？"),
    ("行迹激活",         "星穹列车，力量提升",     "dialogue", "又练了新招？"),
    ("队伍编成•",        "队伍集结，准备出发",     "dialogue", "准备好了吗？"),
    ("战斗开始•弱点击破", "即将进入战斗，气氛紧张", "action",   "敌人挡在了面前"),
    ("战斗开始•危险预警", "前方出现不明生物",     "dialogue", "小心！"),
    ("回合开始•",        "战斗中，敌人逼近",       "action",   "敌人挡在了面前"),
    ("回合待机",         "战斗中，敌人徘徊",       "action",   "敌人迟迟没有动作"),
    ("强化普攻•",        "战斗中，你冲上前去",     "action",   "敌人挡在了面前"),
    ("普攻",             "战斗中，你冲上前去",     "action",   "敌人挡在了面前"),
    ("战技•",            "战斗中，你准备出手",     "action",   "敌人挡在了面前"),
    ("轻受击",           "战斗中，你被击中了",     "action",   "敌人的攻击擦到了你"),
    ("重受击",           "战斗中，你受了重伤",     "action",   "敌人的重击打在你身上"),
    ("终结技•激活",      "战斗中，时机到了",       "action",   "敌人露出了破绽"),
    ("终结技•施放",      "战斗中，最后一击",       "action",   "敌人露出了破绽"),
    ("天赋•",            "战斗中，形势好转",       "action",   "局势开始向你倾斜"),
    ("无法战斗",         "战斗中，你倒下了",       "action",   "你倒在了地上"),
    ("重回战斗",         "战斗中，你重新站了起来", "action",   "你撑着站了起来"),
    ("回复生命",         "战斗中，你被治疗了",     "action",   "开拓者替你挡下了攻击"),
    ("秘技",             "战斗前，你先行出手",     "action",   "你悄悄摸到了敌人身后"),
    ("战斗胜利",         "刚打完一场硬仗",         "dialogue", "总算结束了。"),
    ("开启贵重战利品",   "战斗后，打开战利品箱",   "action",   "开拓者打开了那个沉甸甸的箱子"),
    ("开启战利品•",      "战斗后，打开战利品箱",   "action",   "开拓者打开了战利品箱"),
    ("解谜成功•",        "解开了一个机关谜题",     "dialogue", "成功了！"),
    ("发现敌方目标",     "前方出现不明生物",       "dialogue", "前面有东西。"),
    ("返回城镇",         "回到城镇，稍作休整",     "dialogue", "总算回来了。"),
]


def voice_synth_for(section):
    for pre, scene, mode, line in VOICE_SYNTH:
        if section and section.startswith(pre):
            return scene, mode, line
    return ("星穹列车车厢，闲聊", "dialogue", "嗯。")


# ============================================================================
# 从 by_mission 构建 turn 序列
# ============================================================================
def in_parens(text):
    t = text.strip()
    return f"（{t}）" if not (t.startswith("（") and t.endswith("）")) else t


def narration_ok(text):
    """该旁白能否作为「动作输入」。排除系统提示与过短/过长项。"""
    t = (text or "").strip()
    if not (ACTION_MIN_CHARS <= len(t) <= ACTION_MAX_CHARS):
        return False
    if UI_BLOCK_RE.search(t):
        return False
    if "气势值" in t:
        return False
    return True


def build_turns(rows):
    """把 mission 的原始行折叠成 turn 列表。

    关键：连续的 kind=='choice' 行是**同一处的并列选项**，不是先后对白，
    必须折叠成一个 turn，否则会构造出语义错乱的上下文。
    """
    rows = sorted(rows, key=lambda r: r.get("pos", 0))
    turns, i = [], 0
    while i < len(rows):
        r = rows[i]
        if r.get("lang") not in (None, "zh"):
            i += 1
            continue
        if r.get("kind") == "choice":                      # 折叠并列选项
            j, alts = i, []
            while j < len(rows) and rows[j].get("kind") == "choice":
                t = norm_text((rows[j].get("text") or "").strip())
                if t:
                    alts.append(t)
                j += 1
            if alts:
                turns.append({"speaker": rows[i].get("speaker") or PLAYER,
                              "text": alts[0], "is_choice": True, "is_action": False,
                              "section": rows[i].get("section"), "region": rows[i].get("region"),
                              "channel": rows[i].get("channel"), "kind": "choice"})
            i = j
            continue
        spk = (r.get("speaker") or "").strip()
        txt = norm_text((r.get("text") or "").strip())
        if not txt:
            i += 1
            continue
        if spk in ALIASES:
            turns.append({"speaker": CHAR, "text": txt, "is_choice": False, "is_action": False,
                          "section": r.get("section"), "region": r.get("region"),
                          "channel": r.get("channel"), "kind": r.get("kind")})
        elif spk == "":
            turns.append({"speaker": None, "text": txt, "is_choice": False, "is_action": True,
                          "section": r.get("section"), "region": r.get("region"),
                          "channel": r.get("channel"), "kind": "narration"})
        else:
            turns.append({"speaker": spk, "text": txt, "is_choice": False, "is_action": False,
                          "section": r.get("section"), "region": r.get("region"),
                          "channel": r.get("channel"), "kind": r.get("kind")})
        i += 1
    return turns


def find_runs(turns):
    """三月七连续发言游程 [start, end)。"""
    runs, i = [], 0
    while i < len(turns):
        if turns[i]["speaker"] == CHAR:
            j, texts = i, []
            while j < len(turns) and turns[j]["speaker"] == CHAR:
                texts.append(turns[j]["text"])
                j += 1
            runs.append((i, j, texts))
            i = j
        else:
            i += 1
    return runs


def char_turn(text, ref):
    return {"speaker": CHAR, "text": text, "is_choice": False, "is_action": False,
            "section": ref.get("section"), "region": ref.get("region"),
            "channel": ref.get("channel"), "kind": "dialogue"}


def collect_context(turns, s, k, allowed_choice=1):
    """从 s-1 往回取最多 k 个 turn，含至多 allowed_choice 个 choice turn（分支安全）。"""
    return tail_ctx(turns[:s], k, allowed_choice)


def tail_ctx(pre, k, allowed_choice=1):
    """从 pre 末尾往回取最多 k 个 turn。

    分支安全：一个样本的上下文里最多出现 1 个 choice turn —— 因为语料把同一处的
    并列选项按顺序平铺，若把两个选项同时放进上下文，会拼出语义错乱的对话。
    散文旁白（>NARRATION_MAX_CHARS）作为上下文边界。
    """
    ctx, used_choice, i = [], 0, len(pre) - 1
    while i >= 0 and len(ctx) < k:
        t = pre[i]
        if t["is_action"] and len(t["text"]) > NARRATION_MAX_CHARS:
            break
        if t["is_choice"]:
            if used_choice >= allowed_choice:
                break
            used_choice += 1
        ctx.append(t)
        i -= 1
    ctx.reverse()
    return ctx


def render_line(t):
    return in_parens(t["text"]) if t["is_action"] else f"{t['speaker']}：{t['text']}"


def narration_renderable(text):
    """旁白能否作为上下文里的「（动作）」出现。剔除系统/UI 标记。"""
    t = (text or "").strip()
    return len(t) >= ACTION_MIN_CHARS and not UI_BLOCK_RE.search(t)


def repeats_context(asst, ctx, thr=0.55, minlen=10, lcs_min=6):
    """assistant 是否在复读上下文里已有的台词。

    语料按分支平铺，同一句台词常被记录两次（措辞略有差异）。若不过滤，
    模型会学到"把输入复述一遍"——这比输出过短更糟。
    两条判据：整句相似度、以及"最长公共子串"（抓改写式复读，如
    「我还得多拍几张纪念照」→「我还想多拍几张纪念照呢」）。
    """
    alines = [x.strip() for x in asst.split("\n") if x.strip()]
    clines = [t["text"].strip() for t in ctx if t["text"] and t["text"].strip()]
    for a in alines:
        if len(a) < minlen:
            continue
        for c in clines:
            if len(c) < minlen:
                continue
            if a in c or c in a:
                return True
            if SequenceMatcher(None, a, c).ratio() >= thr:
                return True
            m = SequenceMatcher(None, a, c).find_longest_match(0, len(a), 0, len(c))
            if m.size >= lcs_min:
                return True
    return False


def make_user_content(variant, scene, ctx):
    if variant == "scene_dialogue":
        # 剔掉上下文中间的 UI/系统旁白（保留末轮，它决定输入形态）
        keep = [t for i, t in enumerate(ctx)
                if i == len(ctx) - 1 or not t["is_action"] or narration_renderable(t["text"])]
        ctx = keep or ctx
    body = "\n".join(render_line(t) for t in ctx)
    if variant in ("scene_dialogue", "scene_action"):
        return f"【场景：{scene}】\n{body}"
    if variant == "bare_action":
        return body
    if variant == "bare_line":
        return in_parens(ctx[-1]["text"]) if ctx[-1]["is_action"] else ctx[-1]["text"]
    raise ValueError(variant)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--model", default="/home/user/npc-models/MiniCPM5-1B")
    ap.add_argument("--target", type=int, default=1450, help="目标样本总数")
    ap.add_argument("--max-merge-lines", type=int, default=2,
                    help="③ 一段 assistant 最多合并几轮连续台词（calibrate.py 标定为 2）")
    ap.add_argument("--max-pieces-per-run", type=int, default=3,
                    help="一个长游程最多切成几个样本，防止一段独白刷出十几条")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--min-val", type=int, default=60)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--max-total-tokens", type=int, default=400)
    ap.add_argument("--single-min-chars", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20251005)
    args = ap.parse_args()

    ntok, tk = load_tokenizer(args.model)
    log(f"tokenizer: {'OK' if tk else 'fallback'}  target={args.target} "
        f"cap={args.max_merge_lines}")

    bm = os.path.join(args.corpus, "by_mission")
    files = sorted(f for f in os.listdir(bm) if f.endswith(".jsonl"))
    log(f"by_mission 文件数: {len(files)}")

    # ---------------- 候选生成：游程 → 连续片段 ----------------
    cands = []
    src = collections.Counter()
    cap = args.max_merge_lines

    for fn in files:
        mission = fn[:-6]
        turns = build_turns(list(iter_jsonl(os.path.join(bm, fn))))
        for (s, e, raw_texts) in find_runs(turns):
            texts = [t.strip() for t in raw_texts if t.strip()]
            if not texts:
                continue
            single_run = (len(texts) == 1)
            pieces = [texts[i:i + cap] for i in range(0, len(texts), cap)]
            if len(pieces) > args.max_pieces_per_run:
                src["drop_piece_overflow"] += len(pieces) - args.max_pieces_per_run
                pieces = pieces[:args.max_pieces_per_run]

            prefix = []                                   # 前面片段 → 上下文
            for piece in pieces:
                asst = "\n".join(piece)
                n_ok = ntok(asst) >= args.min_tokens
                # ③ 单行游程只保留较长台词；多行片段一律保留（它是续写切片）
                len_ok = (not single_run) or len(piece[0]) >= args.single_min_chars
                if n_ok and len_ok:
                    pre = turns[:s] + prefix               # 目标台词之前的完整连续历史
                    if not pre:                            # 游程在 mission 开头，无上下文可用
                        src["drop_no_ctx"] += 1
                    else:
                        cands.append({
                            "kind_src": "dialogue", "mission": mission,
                            "assistant": asst, "nlines": len(piece),
                            "pre": pre, "meta": turns[s],
                            "piece_idx": len(prefix),
                        })
                        src["kept_piece"] += 1
                        if len(prefix):
                            src["kept_piece_continuation"] += 1
                else:
                    src["drop_piece"] += 1
                prefix = prefix + [char_turn(t, turns[s]) for t in piece]
    log(f"对话候选: {len(cands)}  {dict(src)}")

    # ---------------- 角色语音层 ----------------
    voice_rows = [r for r in iter_jsonl(os.path.join(args.corpus, "dialogue_full.jsonl"))
                  if r.get("speaker") in ALIASES and r.get("kind") == "voice"]
    voice_cands = []
    for r in voice_rows:
        sec = (r.get("section") or "").strip()
        txt = (r.get("text") or "").strip()
        if not txt or ntok(txt) < 2:                      # 战斗语音天然很短
            continue
        scene, mode, line = voice_synth_for(sec)
        grp = sec.split("•")[0] or "misc"
        if mode == "action":
            t = {"speaker": None, "text": line.strip("（）"), "is_action": True, "is_choice": False,
                 "section": sec, "region": "", "channel": "voice", "kind": "narration"}
        else:
            t = {"speaker": PLAYER, "text": line, "is_action": False, "is_choice": False,
                 "section": sec, "region": "", "channel": "voice", "kind": "dialogue"}
        voice_cands.append({"kind_src": "voice", "mission": f"__voice__/{grp}",
                            "assistant": txt, "nlines": 1, "pre": [t],
                            "scene": scene, "synth_ctx": True, "meta": t})
    log(f"角色语音候选: {len(voice_cands)}")

    # ---------------- 预算：片段 → 语音 → 长单行(按 region 分层补足) ----------------
    multi = [c for c in cands if c["nlines"] >= 2]
    single = [c for c in cands if c["nlines"] == 1]
    budget = max(0, args.target - len(multi) - len(voice_cands))
    log(f"多行片段={len(multi)} 单行片段={len(single)} 语音={len(voice_cands)} "
        f"→ 单行预算={budget}")

    by_region = collections.defaultdict(list)
    for c in single:
        by_region[location_of(c["meta"].get("region")) or "未知"].append(c)
    # ③ 地区配额内优先取更长的单行台词（长度偏置，README 中说明）
    for k in by_region:
        by_region[k].sort(key=lambda c: (-ntok(c["assistant"]), md5i("s|" + c["assistant"])))

    picked_single, total_single = [], sum(len(v) for v in by_region.values())
    if budget > 0 and total_single:
        for k, v in sorted(by_region.items()):
            picked_single.extend(v[:int(round(budget * len(v) / total_single))])
        taken = set(id(c) for c in picked_single)
        rest = [c for k, v in sorted(by_region.items()) for c in v if id(c) not in taken]
        picked_single.extend(rest[:max(0, budget - len(picked_single))])

    selected = multi + voice_cands + picked_single
    log(f"选中合计={len(selected)}（多行 {len(multi)} / 语音 {len(voice_cands)} / "
        f"单行 {len(picked_single)}）")

    # ---------------- 变体分配 ----------------
    def pick_ctx(c, variant, k):
        """返回该样本实际使用的上下文；该形态不可用则 None。

        `pre` 是目标台词之前的**连续**历史，其任意以末尾为终点的后缀都紧邻目标台词，
        所以「在 pre 里回溯到最后一个合适的动作旁白，并取以它结尾的窗口」是合法的
        短上下文切片（不是拼接）。动作旁白在近处窗口里很稀少，因此需要在整个 pre 里找。
        """
        pre = c["pre"]
        if variant in ("bare_action", "scene_action"):
            for i in range(len(pre) - 1, -1, -1):
                t = pre[i]
                if t["is_action"] and narration_ok(t["text"]):
                    if variant == "bare_action":
                        return [t]                          # 推理场景①：纯动作
                    return tail_ctx(pre[:i + 1], k)          # 推理场景⑥：场景 + 动作
            return None
        if variant == "bare_line":
            # 推理里的裸输入是**玩家/旁白说的一句话**，不是三月七自己的半句话，
            # 也不是动作（动作有 bare_action / scene_action 两种专属形态）。
            last = pre[-1]
            return [last] if (last["speaker"] != CHAR and not last["is_action"]) else None
        return tail_ctx(pre, k)

    total = len(selected)
    quota = {"bare_action": int(total * 0.15), "scene_action": int(total * 0.10),
             "bare_line": int(total * 0.18)}
    order = sorted(range(total), key=lambda i: md5i("v|%d|%s" % (i, selected[i]["assistant"][:40])))
    used, ctxs, need = [None] * total, [None] * total, dict(quota)
    for i in order:                                         # 阶段1：动作类形态
        c = selected[i]
        k = max(1, 2 + md5i("k|" + c["assistant"], 3))
        for v in ("bare_action", "scene_action"):
            if need[v] > 0:
                cx = pick_ctx(c, v, k)
                if cx:
                    used[i], ctxs[i], need[v] = v, cx, need[v] - 1
                    break
    rest = [i for i in range(total) if used[i] is None]      # 阶段2：结构化 / 裸输入
    want_bl = int(len(rest) * 0.22)
    for i in rest:
        c = selected[i]
        k = max(1, 2 + md5i("k|" + c["assistant"], 3))
        cx = pick_ctx(c, "bare_line", k)
        if want_bl > 0 and cx and md5i("b|%s|%d" % (c["assistant"], i), 100) < 45:
            used[i], ctxs[i], want_bl = "bare_line", cx, want_bl - 1
        else:
            used[i], ctxs[i] = "scene_dialogue", pick_ctx(c, "scene_dialogue", k)

    # ---------------- 组装 ----------------
    samples, fmt = [], collections.Counter()
    for i, c in enumerate(selected):
        variant, ctx = used[i], ctxs[i]
        if not ctx:
            variant = "scene_dialogue"
            ctx = tail_ctx(c["pre"], max(1, 2 + md5i("k|" + c["assistant"], 3)))
        scene = c.get("scene") or build_scene(
            c["meta"], " ".join(t["text"] for t in ctx),
            c["meta"].get("channel"), c["meta"].get("section"))
        user, asst = make_user_content(variant, scene, ctx), c["assistant"]
        u_tok, a_tok = ntok(user), ntok(asst)
        if u_tok + a_tok > args.max_total_tokens:
            fmt["drop_too_long"] += 1
            continue
        # 语料会把同一句台词在不同分支下重复记录，导致 assistant 复读上下文里已有的台词。
        # 这种样本会教模型"复读输入"，必须剔除。
        if repeats_context(asst, ctx):
            fmt["drop_self_repeat"] += 1
            continue
        samples.append({
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": asst}],
            "character": CHAR,
            "task": "voice_synth_context" if c.get("synth_ctx") else "story_dialogue",
            "source": "corpus:character_voice" if c["kind_src"] == "voice" else "corpus:by_mission",
            "mission": c["mission"],
            "kind": c["kind_src"],
            "channel": c["meta"].get("channel") or "mission",
            "region": c["meta"].get("region") or "",
            "section": c["meta"].get("section") or "" if section_ok(c["meta"].get("section")) else "",
            "raw_section": c["meta"].get("section") or "",
            "scene": scene,
            "variant": variant,
            "ctx_turns": len(ctx),
            "has_choice_ctx": int(any(t["is_choice"] for t in ctx)),
            "ctx_has_self": int(any(t["speaker"] == CHAR for t in ctx)),
            "assistant_lines": asst.count("\n") + 1,
            "assistant_chars": len(asst),
            "user_tokens": u_tok,
            "assistant_tokens": a_tok,
            "synthetic_context": int(bool(c.get("synth_ctx"))),
        })
        fmt["variant_" + variant] += 1

    # ---------------- 去重 ----------------
    def dedup(rows, keyf, tag):
        seen, out, n = set(), [], 0
        for s in rows:
            k = keyf(s)
            if k in seen:
                n += 1
                continue
            seen.add(k)
            out.append(s)
        log(f"去重[{tag}]: 丢弃 {n} 条，剩 {len(out)}")
        return out, n

    samples, d1 = dedup(samples, lambda s: (s["messages"][1]["content"], s["messages"][2]["content"]), "完全相同 user+assistant")
    final, d2 = dedup(samples, lambda s: (s["mission"], s["messages"][2]["content"]), "同 mission 内重复台词")

    # ---------------- 按 mission 切分 ----------------
    by_m = collections.defaultdict(list)
    for s in final:
        by_m[s["mission"]].append(s)
    missions = sorted(by_m)
    rng = random.Random(args.seed + 1)
    rng.shuffle(missions)
    want_val = max(args.min_val, int(round(len(final) * args.val_ratio)))
    # 先排除"超大 mission"，否则 2 个 mission 就能填满 val，val_loss 只反映单一剧情线
    cap_m = max(3, want_val // 8)
    pool_m = [m for m in missions if len(by_m[m]) <= cap_m] or missions[:]
    val_m, cur, cand = [], 0, pool_m[:]
    while cur < want_val and cand:
        best = min(cand, key=lambda m: (abs(cur + len(by_m[m]) - want_val), md5i("v|" + m)))
        val_m.append(best)
        cur += len(by_m[best])
        cand.remove(best)
    if cur < want_val:
        chosen = set(val_m)
        for m in sorted([m for m in missions if m not in chosen],
                        key=lambda m: (len(by_m[m]), md5i("w|" + m))):
            if cur >= want_val:
                break
            val_m.append(m)
            cur += len(by_m[m])
    val_set = set(val_m)
    train = [s for m in missions if m not in val_set for s in by_m[m]]
    val = [s for m in val_m for s in by_m[m]]
    overlap = set(s["mission"] for s in train) & set(s["mission"] for s in val)
    log(f"split: train={len(train)} ({len(set(s['mission'] for s in train))} mission)  "
        f"val={len(val)} ({len(val_set)} mission)  重叠={len(overlap)}")

    # ---------------- 写盘 ----------------
    os.makedirs(args.out, exist_ok=True)
    rng3 = random.Random(args.seed + 2)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        rows = list(rows)
        rng3.shuffle(rows)
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"写出 {p}: {len(rows)} 条")

    def dist(rows):
        a = [s["assistant_tokens"] for s in rows]
        u = [s["user_tokens"] for s in rows]
        c = [s["assistant_chars"] for s in rows]
        ss = lambda v: sorted(v)

        def q(v, p):
            v = ss(v)
            return v[min(len(v) - 1, int(len(v) * p))] if v else 0
        return {
            "n": len(rows),
            "n_missions": len(set(s["mission"] for s in rows)),
            "assistant_tokens": {"min": min(a) if a else 0, "p50": q(a, .5), "p90": q(a, .9),
                                 "max": max(a) if a else 0,
                                 "mean": round(sum(a) / len(a), 1) if a else 0},
            "assistant_chars": {"min": min(c) if c else 0, "p50": q(c, .5), "p90": q(c, .9),
                                "max": max(c) if c else 0,
                                "mean": round(sum(c) / len(c), 1) if c else 0},
            "user_tokens": {"min": min(u) if u else 0, "p50": q(u, .5), "p90": q(u, .9),
                            "max": max(u) if u else 0,
                            "mean": round(sum(u) / len(u), 1) if u else 0},
            "variant": dict(collections.Counter(s["variant"] for s in rows)),
            "kind": dict(collections.Counter(s["kind"] for s in rows)),
            "channel": dict(collections.Counter(s["channel"] for s in rows)),
            "region": dict(collections.Counter(s["region"] for s in rows).most_common()),
            "assistant_lines": dict(collections.Counter(s["assistant_lines"] for s in rows)),
            "synthetic_context": sum(s["synthetic_context"] for s in rows),
        }

    stats = {
        "source": {"corpus": args.corpus, "by_mission_files": len(files),
                   "voice_rows": len(voice_rows)},
        "params": vars(args),
        "candidates": dict(src),
        "variant_assign": dict(fmt),
        "placeholder_normalized": dict(_norm_hits),
        "duplicates_removed": {"exact_pair": d1, "same_mission_same_line": d2},
        "total_before_split": len(final),
        "leakage_mission_overlap": len(overlap),
        "train": dist(train), "val": dist(val),
    }
    with open(os.path.join(args.out, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"train assistant token mean={stats['train']['assistant_tokens']['mean']} "
        f"p50={stats['train']['assistant_tokens']['p50']} "
        f"chars_mean={stats['train']['assistant_chars']['mean']}")
    log(f"val   assistant token mean={stats['val']['assistant_tokens']['mean']}")
    log("变体:", stats["train"]["variant"])


if __name__ == "__main__":
    main()
