#!/usr/bin/env python3
"""Fixed llava->text_encoder preprocess for transformers>=5.x (LLM now at model.model.language_model)."""
import argparse
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    a = ap.parse_args()
    processor = AutoProcessor.from_pretrained(a.input_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        a.input_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    llm = getattr(model, "language_model", None)
    if llm is None and hasattr(model, "model"):
        llm = getattr(model.model, "language_model", None)
    if llm is None:
        raise AttributeError("cannot find language_model in "
                             + str([n for n, _ in model.named_children()]))
    llm.save_pretrained(a.output_dir)
    processor.tokenizer.save_pretrained(a.output_dir)
    print("saved to", a.output_dir)


if __name__ == "__main__":
    main()
