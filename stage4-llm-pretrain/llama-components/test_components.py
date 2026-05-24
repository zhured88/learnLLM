"""
LLaMA 组件正确性测试

验证每一项改进的正确性:
  1. RMSNorm: 数值与 LayerNorm 的一致性（归一化后 std ≈ 1）
  2. SwiGLU: 输出 shape 和参数量正确
  3. RoPE: 相对位置性质
  4. GQA: 在 n_kv_head=n_head 时与 MHA 等价
  5. LLaMA Block: 前向传播不崩溃，输出 shape 正确
"""

import torch

from rms_norm import RMSNorm
from swiglu import SwiGLU, GELU_MLP, count_params
from rope import RotaryPositionEmbedding
from gqa import GroupedQueryAttention, MultiHeadAttention
from llama_block import LLaMABlock


def test_rms_norm():
    """测试 RMSNorm 归一化效果。"""
    print("1. RMSNorm 测试...", end=" ")
    x = torch.randn(4, 8, 16)
    rms = RMSNorm(16)
    y = rms(x)

    # 归一化后 std ≈ 1（RMSNorm 不做零均值，所以检查 rms）
    rms_val = torch.sqrt(torch.mean(y.float() ** 2, dim=-1))
    assert torch.allclose(rms_val, torch.ones_like(rms_val), atol=0.3), \
        f"RMSNorm 输出 RMS 不为 1: {rms_val}"

    # gamma 参数生效
    rms.gamma.data = torch.full((16,), 2.0)
    y2 = rms(x)
    assert y2.std(dim=-1).mean() > 1.5, "gamma 缩放未生效"

    print("PASSED")


def test_swiglu():
    """测试 SwiGLU 输出 shape 和参数量。"""
    print("2. SwiGLU 测试...", end=" ")
    dim = 256
    swiglu = SwiGLU(dim)
    gelu_mlp = GELU_MLP(dim)

    x = torch.randn(2, 10, dim)
    y = swiglu(x)

    # 输出 shape 正确
    assert y.shape == (2, 10, dim), f"输出 shape 错误: {y.shape}"

    # 参数量接近
    sp = count_params(swiglu)
    gp = count_params(gelu_mlp)
    assert 0.9 < sp / gp < 1.3, f"SwiGLU/GELU 参数量比异常: {sp/gp:.2f}"

    print(f"PASSED (SwiGLU params={sp:,}, GELU params={gp:,}, ratio={sp/gp:.2f})")


def test_rope():
    """测试 RoPE 相对位置性质。"""
    print("3. RoPE 测试...", end=" ")
    d_k = 64
    rope = RotaryPositionEmbedding(d_k, max_seq_len=128)

    # 构造: 相同内容在不同位置，相对位置相同，attention score 应相同
    q_content = torch.randn(1, 1, 1, d_k)
    k_content = torch.randn(1, 1, 1, d_k)

    T = 8
    q_seq = torch.zeros(1, 1, T, d_k)
    k_seq = torch.zeros(1, 1, T, d_k)
    q_seq[:, :, 0, :] = q_content[:, :, 0, :]
    q_seq[:, :, 4, :] = q_content[:, :, 0, :]
    k_seq[:, :, 2, :] = k_content[:, :, 0, :]
    k_seq[:, :, 6, :] = k_content[:, :, 0, :]

    q_rot, k_rot = rope(q_seq, k_seq, seq_len=T)

    score_0_2 = (q_rot[0, 0, 0] * k_rot[0, 0, 2]).sum()
    score_4_6 = (q_rot[0, 0, 4] * k_rot[0, 0, 6]).sum()

    # 相对位置差相同(2), attention score 应该相近
    diff = abs(score_0_2 - score_4_6)
    assert diff < 1e-4, f"RoPE 相对位置性质不满足: diff={diff:.8f}"

    print(f"PASSED (diff={diff:.8f})")


def test_gqa_equivalence():
    """测试 GQA 在 n_kv_head=n_head 时退化为 MHA。"""
    print("4. GQA 等价性测试...", end=" ")
    dim, n_head = 256, 4
    T = 8

    # 创建 GQA(n_kv=4) 和 MHA，手动对齐权重
    gqa = GroupedQueryAttention(dim, n_head, n_kv_head=n_head)
    mha = MultiHeadAttention(dim, n_head)

    # 复制权重
    mha.q_proj.weight.data = gqa.q_proj.weight.data.clone()
    mha.k_proj.weight.data = gqa.k_proj.weight.data.clone()
    mha.v_proj.weight.data = gqa.v_proj.weight.data.clone()
    mha.o_proj.weight.data = gqa.o_proj.weight.data.clone()

    x = torch.randn(2, T, dim)
    mask = torch.tril(torch.ones(1, 1, T, T))

    with torch.no_grad():
        y_gqa = gqa(x, mask=mask)
        y_mha = mha(x, mask=mask)

    max_diff = (y_gqa - y_mha).abs().max().item()
    assert max_diff < 1e-5, f"GQA 和 MHA 输出不一致: diff={max_diff:.8f}"

    print(f"PASSED (max diff={max_diff:.8f})")


def test_gqa_kv_ratio():
    """测试 GQA KV 头数为 Q 头数的 1/4。"""
    print("5. GQA KV分组测试...", end=" ")
    dim, n_head, n_kv_head = 256, 8, 2
    gqa = GroupedQueryAttention(dim, n_head, n_kv_head)

    x = torch.randn(2, 16, dim)
    mask = torch.tril(torch.ones(1, 1, 16, 16))
    y = gqa(x, mask=mask)

    assert y.shape == (2, 16, dim), f"输出 shape 错误: {y.shape}"

    # 验证 K/V 投影维度
    assert gqa.k_proj.weight.shape[0] == n_kv_head * (dim // n_head), \
        f"K 投影维度错误: {gqa.k_proj.weight.shape}"
    assert gqa.v_proj.weight.shape[0] == n_kv_head * (dim // n_head), \
        f"V 投影维度错误: {gqa.v_proj.weight.shape}"

    print("PASSED")


def test_llama_block():
    """测试完整 LLaMA Block 前向传播。"""
    print("6. LLaMA Block 测试...", end=" ")
    dim, n_head, n_kv_head = 256, 8, 4
    block = LLaMABlock(dim, n_head, n_kv_head, max_seq_len=64)

    x = torch.randn(2, 16, dim)
    mask = torch.tril(torch.ones(1, 1, 16, 16))
    y = block(x, mask)

    assert y.shape == x.shape, f"输出 shape 不匹配: {y.shape} != {x.shape}"
    assert not torch.isnan(y).any(), "输出含 NaN"
    assert not torch.isinf(y).any(), "输出含 Inf"

    print(f"PASSED (params={sum(p.numel() for p in block.parameters()):,})")


if __name__ == "__main__":
    print("=" * 55)
    print("Stage 4 · LLaMA 组件正确性测试")
    print("=" * 55)

    test_rms_norm()
    test_swiglu()
    test_rope()
    test_gqa_equivalence()
    test_gqa_kv_ratio()
    test_llama_block()

    print("\n" + "=" * 55)
    print("全部测试通过!")
    print("=" * 55)
