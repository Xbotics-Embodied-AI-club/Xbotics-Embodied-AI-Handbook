from __future__ import annotations

# unsloth 必须在 trl / peft / transformers / datasets 之前导入，否则它对 GRPO 训练的兼容补丁不会生效。
from unsloth import FastVisionModel
from unsloth_zoo.utils import Version

import json
import os
import re
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
import lightning as L
from peft import PeftModel
import qwen_vl_utils
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb


def default_paths() -> dict[str, str]:
    """路径只读环境变量。.env/.envrc 负责加载，代码里不写默认值。"""

    # 结果摘要随课程代码走，训练产物随数据根走。
    result_dir = Path(__file__).resolve().parents[1] / "result" / "2_1_grpo_vlm_counting"
    trained_root = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "xbotics_rl_grpo_vlm"
    return {
        "hf_home": os.environ["HF_HOME"],
        "trained_root": str(trained_root),
        "result_dir": str(result_dir),
    }


def extract_answer(text: str) -> str | None:
    """从 <answer>...</answer> 标签里取出最终计数。"""

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if re.fullmatch(r"[+-]?\d+", stripped):
        return str(int(stripped))
    return stripped.lower()


def format_reward(completion: str) -> float:
    return 1.0 if extract_answer(completion) is not None else 0.0


def correctness_reward(completion: str, target: str) -> float:
    predicted = normalize_answer(extract_answer(completion))
    expected = normalize_answer(extract_answer(target) or target)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def build_run_dir(run_name: str | None = None) -> Path:
    """生成本次训练目录：adapter、logs、eval、wandb 都放在同一个 run 下面。"""

    paths = default_paths()
    name = run_name or "grpo-qwen25vl3b-clevr-" + datetime.now().strftime("%Y%m%d-%H%M")
    return Path(paths["trained_root"]) / name


def training_prompt(problem: str) -> list[dict]:
    # 训练时给模型的任务很窄：看图、数数、只在标签里写最终答案。
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{problem}\nReturn only the final count inside <answer> and </answer>."},
            ],
        }
    ]


def tagged_answer(value) -> str:
    # 标准答案也统一成标签格式，奖励函数就能直接比较最终计数。
    text = str(value).strip()
    if text.lower().startswith("<answer>"):
        return text
    return f"<answer> {text} </answer>"


def first_present(example: dict, field_names: list[str]):
    for field_name in field_names:
        if field_name in example and example[field_name] is not None:
            return example[field_name]
    raise KeyError(f"missing any of fields: {field_names}")


def prepare_training_example(example: dict) -> dict:
    """把 CLEVR 训练样本整理成 GRPOTrainer 需要的 image、prompt、answer。"""

    # 训练图像统一成 RGB + 512x512，减少视觉输入形状上的干扰。
    image = example["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((512, 512))
    return {
        "prompt": training_prompt(example["problem"]),
        "image": image,
        "answer": example["solution"],
    }


def prepare_counting_example(example: dict, index: int) -> dict:
    """把不同计数字段名统一成 predict/test 都能读懂的格式。"""

    # 不同公开数据集字段名不完全一致，这里只做一次最小统一。
    image = first_present(example, ["image", "img", "picture"])
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")
    return {
        "index": index,
        "image": image,
        "problem": str(first_present(example, ["problem", "question", "query", "prompt"])),
        "target": tagged_answer(first_present(example, ["solution", "answer", "target", "label"])),
    }


def describe_image(image) -> str:
    """把图像尺寸和颜色模式压成一行，方便训练前快速检查数据。"""

    width, height = image.size
    return f"{width}x{height}, {image.mode}"


class ClevrCountingDataset(Dataset):
    """普通 PyTorch Dataset：每次返回一张图、一句问题、一个标准答案。"""

    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]

    def with_chat_template(self, tokenizer) -> "ClevrCountingDataset":
        formatted_examples = []
        for example in self.examples:
            formatted_examples.append(
                {
                    **example,
                    "prompt": tokenizer.apply_chat_template(
                        example["prompt"],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                }
            )
        return ClevrCountingDataset(formatted_examples)


class ClevrCountingData(L.LightningDataModule):
    """LightningDataModule：统一准备训练、预测、测试三种 dataloader。"""

    def __init__(self, train_dataset_id: str, eval_dataset_id: str, train_examples: int, eval_examples: int) -> None:
        super().__init__()
        self.train_dataset_id = train_dataset_id
        self.eval_dataset_id = eval_dataset_id
        self.train_examples = train_examples
        self.eval_examples = eval_examples

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            # 训练只取一个小切片，单卡 GRPO 能在演示时间内看到变化。
            train_split = f"train[:{self.train_examples}]"
            raw_train_dataset = load_dataset(self.train_dataset_id, split=train_split)
            self.train_dataset = ClevrCountingDataset(
                [prepare_training_example(example) for example in raw_train_dataset]
            )

        if stage in (None, "predict", "test"):
            # 评测固定取前 eval_examples 条，训练前后对比才有同一把尺子。
            raw_eval_dataset = load_dataset(self.eval_dataset_id, split="train")
            raw_eval_dataset = raw_eval_dataset.select(range(min(self.eval_examples, len(raw_eval_dataset))))
            self.eval_dataset = ClevrCountingDataset(
                [prepare_counting_example(example, index) for index, example in enumerate(raw_eval_dataset)]
            )

    def preview_rows(self) -> list[dict[str, str | int]]:
        """抽几条训练/评测样本，展示模型到底在学什么。"""

        train_preview = load_dataset(self.train_dataset_id, split="train[:2]")
        eval_preview = load_dataset(self.eval_dataset_id, split="train[:2]")
        rows = []

        for index, example in enumerate(train_preview):
            prepared = prepare_training_example(example)
            rows.append(
                {
                    "split": "train",
                    "dataset": self.train_dataset_id,
                    "used_examples": self.train_examples,
                    "index": index,
                    "image": describe_image(prepared["image"]),
                    "problem": str(example["problem"]),
                    "answer": tagged_answer(example["solution"]),
                }
            )

        for index, example in enumerate(eval_preview):
            prepared = prepare_counting_example(example, index)
            rows.append(
                {
                    "split": "eval",
                    "dataset": self.eval_dataset_id,
                    "used_examples": self.eval_examples,
                    "index": index,
                    "image": describe_image(prepared["image"]),
                    "problem": prepared["problem"],
                    "answer": prepared["target"],
                }
            )

        return rows

    def print_preview(self) -> None:
        """训练前打印数据集样例，让读者先看到图像问答任务本身。"""

        print("Dataset preview")
        for row in self.preview_rows():
            print(f"[{row['split']}] {row['dataset']}  index={row['index']}  image={row['image']}")
            print(f"problem: {row['problem']}")
            print(f"answer: {row['answer']}")
            print()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)

    def predict_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)

    def test_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)


class QwenVLGRPOModel(nn.Module):
    """普通 PyTorch 模型对象：加载 Qwen2.5-VL，挂 LoRA，并生成答案。"""

    def __init__(self, model_id: str, fast_inference: bool) -> None:
        super().__init__()
        self.model_id = model_id
        self.fast_inference = fast_inference

    def load_for_training(self) -> None:
        # 训练时加载 4bit 基座模型，显存主要留给生成和 LoRA 更新。
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=8192,
            load_in_4bit=True,
            fast_inference=self.fast_inference,
            gpu_memory_utilization=0.8,
        )
        # 只训练语言侧 LoRA，视觉塔保持冻结；计数任务的反馈主要改写回答方式。
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
            use_gradient_checkpointing="unsloth",
        )

    def load_for_prediction(self, adapter_dir: Path | None) -> None:
        # 评测时先加载同一个基座，再按需挂上训练得到的 LoRA adapter。
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=8192,
            load_in_4bit=True,
            fast_inference=False,
        )
        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        FastVisionModel.for_inference(self.model)

    def build_trainer(self, train_dataset: ClevrCountingDataset, run_dir: Path):
        if Version("trl") < Version("0.24.0"):
            # 旧版 TRL 需要先把 chat template 展开成字符串。
            train_dataset = train_dataset.with_chat_template(self.tokenizer)

        # 格式奖励：模型必须按 <answer>...</answer> 输出，便于把格式约束和正确性分开观察。
        def formatting_reward_func(completions, **unused):
            return [format_reward(completion) for completion in completions]

        # 正确性奖励：只比较最终计数答案，弱化自然语言解释。
        def correctness_reward_func(completions, answer, **unused):
            return [correctness_reward(completion, target) for completion, target in zip(completions, answer)]

        # GRPO 每个问题采样 4 个回答，同组回答互相比较后更新 LoRA。
        grpo_config = GRPOConfig(
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.001,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            log_completions=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=1024,
            max_completion_length=256,
            max_steps=300,
            save_steps=100,
            max_grad_norm=0.1,
            report_to="wandb",
            output_dir=str(run_dir / "logs"),
            importance_sampling_level="sequence",
            mask_truncated_completions=False,
            loss_type="dr_grpo",
        )
        return GRPOTrainer(
            self.model,
            [formatting_reward_func, correctness_reward_func],
            grpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )

    def save_adapter(self, run_dir: Path) -> None:
        self.model.save_lora(str(run_dir / "adapter"))

    def build_messages(self, image, problem: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f"{problem}\nReturn only the final count inside <answer> and </answer>.",
                    },
                ],
            }
        ]

    def generate_answer(self, image, problem: str, max_new_tokens: int) -> str:
        messages = self.build_messages(image, problem)
        # 图像和文本一起进入 Qwen2.5-VL 的 processor/tokenizer。
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
        inputs = self.tokenizer(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)

        # 评测阶段用确定性生成，训练前后结果更容易直接比较。
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class GRPOLightningModule(L.LightningModule):
    """LightningModule：fit 触发 GRPO 训练，predict/test 触发训练后评测。"""

    def __init__(
        self,
        model_id: str,
        run_dir: Path,
        prediction_name: str,
        adapter_dir: Path | None,
        fast_inference: bool,
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.prediction_name = prediction_name
        self.adapter_dir = adapter_dir
        self.vlm = QwenVLGRPOModel(model_id=model_id, fast_inference=fast_inference)
        self.automatic_optimization = False
        self.has_trained = False
        self.prediction_rows = []
        self.test_rows = []

    def setup(self, stage: str) -> None:
        # 一个 run 目录里放 adapter、日志、评测输出和 W&B 本地文件。
        (self.run_dir / "adapter").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "eval").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "wandb").mkdir(parents=True, exist_ok=True)

        if stage == "fit":
            # fit 阶段加载可训练模型，并把 LightningDataModule 准备好的训练集交给 GRPOTrainer。
            wandb.init(project="rl_class", name=self.run_dir.name, dir=str(self.run_dir / "wandb"))
            self.vlm.load_for_training()
            self.grpo_trainer = self.vlm.build_trainer(self.trainer.datamodule.train_dataset, self.run_dir)
        if stage in ("predict", "test"):
            # predict/test 阶段只做生成，不再创建 GRPOTrainer。
            self.vlm.load_for_prediction(self.adapter_dir)

    def training_step(self, batch, batch_idx):
        if not self.has_trained:
            # 这个 Lightning step 只触发一次，内部由 GRPOTrainer 跑完 300 个 GRPO step。
            self.grpo_trainer.train()
            self.vlm.save_adapter(self.run_dir)
            self.has_trained = True
        return torch.zeros((), device=self.device)

    def predict_step(self, batch, batch_idx):
        rows = [self.predict_one(example) for example in batch]
        self.prediction_rows.extend(rows)
        return rows

    def on_predict_epoch_end(self) -> None:
        output_jsonl = self.run_dir / "eval" / f"{self.prediction_name}_predictions.jsonl"
        output_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in self.prediction_rows) + "\n")

    def test_step(self, batch, batch_idx):
        rows = [self.predict_one(example) for example in batch]
        self.test_rows.extend(rows)
        return score_rows(rows)

    def on_test_epoch_end(self) -> None:
        summary_json = self.run_dir / "eval" / f"{self.prediction_name}_summary.json"
        summary_json.write_text(json.dumps(score_rows(self.test_rows), ensure_ascii=False, indent=2) + "\n")

    def predict_one(self, example: dict) -> dict:
        completion = self.vlm.generate_answer(
            example["image"],
            example["problem"],
            max_new_tokens=64,
        )
        return {
            "index": example["index"],
            "problem": example["problem"],
            "target": example["target"],
            "completion": completion,
            "prediction_name": self.prediction_name,
        }

    def configure_optimizers(self):
        return None


def score_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"count": 0, "accuracy": 0.0, "format": 0.0}

    # accuracy 看答案是否正确，format 看输出是否遵守 <answer> 标签。
    accuracy = sum(correctness_reward(row["completion"], row["target"]) for row in rows) / len(rows)
    format_score = sum(format_reward(row["completion"]) for row in rows) / len(rows)
    return {"count": len(rows), "accuracy": accuracy, "format": format_score}


def main() -> None:
    # 主要修改这一段：fit 训练，predict 写 JSONL，test 写 summary。
    running_stage = "fit"
    run_dir = build_run_dir("grpo-qwen25vl3b-clevr-lightning-demo")
    prediction_name = "adapter"
    adapter_dir = run_dir / "adapter"

    data = ClevrCountingData(
        train_dataset_id="leonardPKU/clevr_cogen_a_train",
        eval_dataset_id="MMInstruction/SuperClevr_Val",
        train_examples=512,
        eval_examples=200,
    )
    data.print_preview()

    model = GRPOLightningModule(
        model_id="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        run_dir=run_dir,
        prediction_name=prediction_name,
        adapter_dir=adapter_dir,
        fast_inference=True,
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
    )

    if running_stage == "fit":
        trainer.fit(model, data)
    if running_stage == "predict":
        trainer.predict(model, data)
    if running_stage == "test":
        trainer.test(model, data)


if __name__ == "__main__":
    main()
