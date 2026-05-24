"""
文本生成 — 从训练好的 LLaMA 模型生成文本

使用方法:
  python generate.py --checkpoint checkpoints/llama_tiny_epoch05.pt --prompt "Once upon a time"
"""

import argparse
import torch

from model import LLaMA, create_model
from tokenizer import get_tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
             repetition_penalty: float = 1.1, device: torch.device = None):
    """生成文本。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Tokenize
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], device=device)

    # 生成
    output_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    # Decode
    generated = tokenizer.decode(output_ids[0].tolist())
    return generated


def main():
    parser = argparse.ArgumentParser(description="LLaMA 文本生成")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径 (.pt)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="起始文本")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="温度 (越低越确定)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K 采样")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling 阈值")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="重复惩罚 (>1 惩罚重复)")
    args = parser.parse_args()

    # 加载检查点
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]
    print(f"模型配置: {cfg}")

    # 创建模型
    model = create_model(
        model_size="base",  # 任意配置，会被覆盖
        vocab_size=cfg["vocab_size"],
        max_seq_len=2048,
    )
    # 覆盖模型参数以匹配配置
    model.n_embd = cfg["n_embd"]
    model.n_layer = cfg["n_layer"]
    model.n_head = cfg["n_head"]
    model.n_kv_head = cfg["n_kv_head"]
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"模型加载成功 (epoch={checkpoint['epoch']}, loss={checkpoint['loss']:.4f})")

    # Tokenizer (简单字符级兜底)
    tokenizer = get_tokenizer()

    # 生成
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}")
    generated = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(generated)


if __name__ == "__main__":
    main()
