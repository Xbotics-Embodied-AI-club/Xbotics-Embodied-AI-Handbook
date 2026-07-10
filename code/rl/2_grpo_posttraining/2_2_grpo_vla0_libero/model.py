"""VLA-0 的采样与解码：动作就是一串数字 token。

VLA-0 把机器人动作离散成整数串（每个整数是一个 bin 编号），让 VLM 像生成文本一样
生成动作。这意味着讲 15.4 里对文本 token 的那套 GRPO，可以几乎原样搬到动作 token 上：
本文件提供「生成动作块」「把数字串解码回连续动作」「复算逐 token logprob」三件事。
"""

import torch
import torch.nn.functional as F
import xgrammar as xgr

from lerobot.policies.vla0_smol.modeling_vla0_smol import VLA0SmolPolicy
from lerobot.utils.constants import OBS_STATE


def load_policy(path):
    """从 checkpoint 目录加载 VLA-0 策略（0.5B，SmolVLM2 底座）。"""
    return VLA0SmolPolicy.from_pretrained(path).to("cuda")


def decode_actions(m, gen, batch):
    """把生成的数字串 token 解码回连续动作块。

    合法输出是 horizon×action_dim 个整数（每个是 bin 编号）；格式不合法的样本
    解码成全零动作（等价于一次无效尝试，靠奖励信号自然淘汰）。
    """
    device = gen.device
    bsz = gen.shape[0]
    valid = (gen != m.eos_token_id) & (gen != m.pad_token_id)
    toks = torch.where(valid, gen, torch.tensor(m.pad_token_id, device=device))
    texts = m.processor.batch_decode(toks, skip_special_tokens=True)
    n_exp = m.action_horizon * m.action_dim
    n_bins = m.config.n_state_bins
    finals = []
    for t in texts:
        a = t.strip().split()
        if len(a) != n_exp or not all(x.isdigit() and 0 <= int(x) < n_bins for x in a):
            finals.append(torch.zeros(n_exp, device=device, dtype=torch.long))
        else:
            finals.append(torch.tensor([int(x) for x in a], device=device))
    disc = torch.stack(finals).reshape(bsz, -1, m.action_dim)
    eps_bin = 1e-6
    bins = torch.linspace(-1.0 - eps_bin, 1.0 + eps_bin, n_bins + 1, device=device)
    centers = 0.5 * (bins[:-1] + bins[1:])
    act = centers[disc.clamp(0, n_bins - 1)]
    if m.config.relative_actions:
        act = act + batch[OBS_STATE].unsqueeze(1)
    return act


@torch.no_grad()
def sample_chunk(m, batch, temperature):
    """让 VLA-0 生成一个动作块。

    temperature 传数值时按该温度随机采样（训练时要探索）；传 None 时贪婪解码
    （评测时要确定性）。xgrammar 约束解码保证输出只可能是数字串。
    返回 (动作块, 生成记录)；记录里保存了完整输入输出 token，训练时用来复算 logprob。
    """
    images = m.prepare_images(batch)
    padded, _ = m.create_input_tokens(states=batch[OBS_STATE], images=images,
                                      lang_text=batch.get("task", ""), actions=None)
    input_len = padded["input_ids"].shape[1]
    proc = xgr.contrib.hf.LogitsProcessor(m.compiled_grammar)
    sampling = {"do_sample": False} if temperature is None else {"do_sample": True, "temperature": temperature}
    out = m.vlm.generate(
        input_ids=padded["input_ids"], attention_mask=padded["attention_mask"],
        pixel_values=padded["pixel_values"], pixel_attention_mask=padded["pixel_attention_mask"],
        use_cache=True, max_new_tokens=m.config.max_decoding_steps,
        num_beams=1, eos_token_id=m.eos_token_id, pad_token_id=m.pad_token_id,
        logits_processor=[proc], return_dict_in_generate=True, output_scores=False,
        **sampling,
    )
    gen = out.sequences[:, input_len:]
    act = decode_actions(m, gen, batch)
    rec = {
        "input_ids": padded["input_ids"].cpu(),
        "attention_mask": padded["attention_mask"].cpu(),
        "pixel_values": padded["pixel_values"].cpu(),
        "pixel_attention_mask": padded["pixel_attention_mask"].cpu(),
        "gen": gen.cpu(),
        "input_len": input_len,
    }
    return act[:, : m.config.chunk_size], rec


def sequence_logprob_tok(m, rec, requires_grad):
    """复算一次生成记录里动作 token 的逐 token logprob，shape [1, T]。

    训练时对当前策略调用（requires_grad=True，梯度从这里流回）；
    对冻结基座调用（requires_grad=False）则得到 KL-to-ref 需要的参考 logprob。
    """
    device = next(m.parameters()).device
    input_ids = rec["input_ids"].to(device)
    gen = rec["gen"].to(device)
    input_len = rec["input_len"]
    valid = (gen != m.eos_token_id) & (gen != m.pad_token_id)
    full_ids = torch.cat([input_ids, gen], dim=1)
    full_mask = torch.cat([rec["attention_mask"].to(device), valid.long()], dim=1)
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        out = m.vlm(input_ids=full_ids, attention_mask=full_mask,
                    pixel_values=rec["pixel_values"].to(device),
                    pixel_attention_mask=rec["pixel_attention_mask"].to(device), use_cache=False)
        pred = out.logits[:, input_len - 1:-1, :].float()
        lp_tok = F.log_softmax(pred, dim=-1).gather(-1, gen.unsqueeze(-1)).squeeze(-1)
    return lp_tok, valid
