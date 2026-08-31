"""用 GRPO 后训练 Qwen2.5-VL，让它学会数清图里有几个物体。

第15讲 4.2 节走读这个文件的奖励函数与训练骨架，4.3 节给出它跑出来的前后对照。

整条流水线按 Lightning 四件套组织：`ClevrCountingDataset` 准备样本、
`ClevrCountingData` 包 dataloader、`QwenVLGRPOModel` 是模型本体、
`GRPOLightningModule` 管训练与评测的生命周期，入口是 `trainer.fit(model, data)`。
和第14讲那三版 RL 的一处不同：组采样发生在 TRL 的 `GRPOTrainer` 内部，
所以 `training_step` 只被调用一次，进去之后一口气跑完 300 个 GRPO step。
"""

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
    """列出本实验会读写的两个根目录，训练前先打出来确认边界。

    路径只从环境变量读，`.env` / `.envrc` 负责加载，代码里不写默认值。

    Returns:
        dict[str, str]: `hf_home` 是模型下载缓存，`trained_root` 是本实验训练产物的落点。
    """

    trained_root = Path(os.environ["DATASETS_ROOT"]) / "models" / "trained" / "xbotics_rl_grpo_vlm"
    return {
        "hf_home": os.environ["HF_HOME"],
        "trained_root": str(trained_root),
    }


def extract_answer(text: str) -> str | None:
    """从 <answer>...</answer> 标签里取出最终计数。

    Args:
        text: 模型生成的整段回答，或标准答案。

    Returns:
        str | None: 标签里的内容；没有合法标签时返回 None——这一步失败，
        格式奖励就是 0，正确性奖励也无从谈起。
    """

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def normalize_answer(value: str | None) -> str | None:
    """把答案归一化后再比较，免得"07"和"7"被判成不同答案。

    Args:
        value: 从标签里抽出来的答案文本，可能为 None。

    Returns:
        str | None: 纯整数走 int 往返去掉前导零和正号，其余转小写；输入为 None 时原样返回。
    """

    if value is None:
        return None
    stripped = value.strip()
    if re.fullmatch(r"[+-]?\d+", stripped):
        return str(int(stripped))
    return stripped.lower()


def format_reward(completion: str) -> float:
    """格式奖励：每条回答都判得动，信号密集，先把答案的位置教会。

    Args:
        completion: 模型生成的一条回答。

    Returns:
        float: 抽得出 <answer> 标签得 1.0，否则 0.0。
    """

    return 1.0 if extract_answer(completion) is not None else 0.0


def correctness_reward(completion: str, target: str) -> float:
    """正确性奖励：只比最终计数，不管模型在标签外写了什么解释。

    抽不出答案直接判 0——所以格式奖励必须先立起来，这条才有的放矢。

    Args:
        completion: 模型生成的一条回答。
        target: 标准答案，允许带标签也允许是裸数字。

    Returns:
        float: 归一化后两者相等得 1.0，否则 0.0。
    """

    predicted = normalize_answer(extract_answer(completion))
    expected = normalize_answer(extract_answer(target) or target)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def build_run_dir(run_name: str | None = None) -> Path:
    """生成本次训练目录：adapter、logs、eval、wandb 都放在同一个 run 下面。

    Args:
        run_name: 指定 run 名；留空则按当前时间生成，便于多次训练互不覆盖。

    Returns:
        Path: 训练产物根下的这一次 run 目录。
    """

    name = run_name or "grpo-qwen25vl3b-clevr-" + datetime.now().strftime("%Y%m%d-%H%M")
    return Path(default_paths()["trained_root"]) / name


def training_prompt(problem: str) -> list[dict]:
    """把任务收得很窄：看图、数数、只在标签里写最终答案。

    窄是有意的——回答越自由，判分脚本越抓不准答案在哪。

    Args:
        problem: 数据集里那句问题，形如"图中有几个红色的金属物体"。

    Returns:
        list[dict]: 一条 chat 格式的消息，图像占位符在前、问题在后。
    """

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
    """把标准答案也套上标签，奖励函数两边就能走同一条抽取逻辑。

    Args:
        value: 数据集给的答案，通常是整数，也可能已经带着标签。

    Returns:
        str: 形如 `<answer> 7 </answer>` 的字符串。
    """

    text = str(value).strip()
    if text.lower().startswith("<answer>"):
        return text
    return f"<answer> {text} </answer>"


def first_present(example: dict, field_names: list[str]):
    """公开数据集的字段名各叫各的，按候选名依次找出第一个能用的。

    Args:
        example: 数据集的一条样本。
        field_names: 按优先级排好的候选字段名。

    Returns:
        样本里第一个存在且非空的字段值。

    Raises:
        KeyError: 候选名一个都没命中，说明这份数据集的格式超出了预期。
    """

    for field_name in field_names:
        if field_name in example and example[field_name] is not None:
            return example[field_name]
    raise KeyError(f"missing any of fields: {field_names}")


def prepare_training_example(example: dict) -> dict:
    """把 CLEVR 训练样本整理成 GRPOTrainer 需要的 image、prompt、answer。

    Args:
        example: `leonardPKU/clevr_cogen_a_train` 的一条原始样本。

    Returns:
        dict: 训练用的三件套。图像统一成 RGB + 512x512，是为了把"视觉输入形状"
        这个变量从对照里排除掉。
    """

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
    """把评测集整理成 predict/test 都能读懂的格式。

    Args:
        example: 评测数据集的一条原始样本。
        index: 样本序号，写进结果里才能定位到具体是哪一题答错了。

    Returns:
        dict: 含 index、image、problem、target 四项。
    """

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
    """把图像尺寸和颜色模式压成一行，方便训练前扫一眼数据。

    Args:
        image: PIL 图像。

    Returns:
        str: 形如 `512x512, RGB`。
    """

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
        """把 prompt 预先展开成字符串，供旧版 TRL 使用。

        Args:
            tokenizer: 带 chat template 的 processor。

        Returns:
            ClevrCountingDataset: prompt 已展开的新数据集，原数据集不动。
        """

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
        """按阶段拉取数据集切片。

        Args:
            stage: Lightning 传入的阶段名（fit / predict / test），None 表示全都准备。
        """

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
        """抽几条训练/评测样本，展示模型到底在学什么。

        Returns:
            list[dict]: 每行含 split、dataset、index、image、problem、answer，可直接打印。
        """

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
        """训练 dataloader。

        Returns:
            DataLoader: batch_size=1 且不打乱——真正的组采样在 GRPOTrainer 内部进行，
            这里只是把训练集递进去。
        """

        return DataLoader(self.train_dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)

    def predict_dataloader(self):
        """预测 dataloader，逐条生成回答并写 JSONL。

        Returns:
            DataLoader: 固定顺序的评测集。
        """

        return DataLoader(self.eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)

    def test_dataloader(self):
        """测试 dataloader，与 predict 用同一套题，保证前后对比是同一把尺子。

        Returns:
            DataLoader: 固定顺序的评测集。
        """

        return DataLoader(self.eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)


class QwenVLGRPOModel(nn.Module):
    """普通 PyTorch 模型对象：加载 Qwen2.5-VL，挂 LoRA，并生成答案。"""

    def __init__(self, model_id: str, fast_inference: bool) -> None:
        super().__init__()
        self.model_id = model_id
        self.fast_inference = fast_inference

    def load_for_training(self) -> None:
        """加载 4bit 基座并挂上 LoRA，显存主要留给生成和 LoRA 更新。

        视觉塔保持冻结、只训语言侧：数数任务的反馈改的是"怎么把答案说出来"，
        不是"怎么看图"。
        """

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=8192,
            load_in_4bit=True,
            fast_inference=self.fast_inference,
            gpu_memory_utilization=0.8,
        )
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
        """加载同一个基座用于评测，按需挂上训练得到的 LoRA adapter。

        Args:
            adapter_dir: adapter 目录；传 None 就评测后训练前的基座，
            两次评测共用这一个入口，前后对比才没有实现差异。
        """

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
        """把两条判分规则和一份 GRPO 配置交给 TRL，组采样与优势计算都由它代劳。

        Args:
            train_dataset: 已准备好的训练集。
            run_dir: 本次 run 的目录，日志写在它下面的 logs/。

        Returns:
            GRPOTrainer: 调用它的 train() 就跑完全部 GRPO step。
        """

        if Version("trl") < Version("0.24.0"):
            # 旧版 TRL 需要先把 chat template 展开成字符串。
            train_dataset = train_dataset.with_chat_template(self.tokenizer)

        def formatting_reward_func(completions, **unused):
            """格式奖励，逐条打分；把格式约束和正确性分开，才看得出模型先学会了哪一件。

            Args:
                completions: GRPOTrainer 一次送来同一道题的一整组回答。

            Returns:
                list[float]: 与 completions 等长的奖励列表。
            """
            return [format_reward(completion) for completion in completions]

        def correctness_reward_func(completions, answer, **unused):
            """正确性奖励，逐条打分；只比最终计数，模型怎么解释不影响得分。

            Args:
                completions: 同一道题的一整组回答。
                answer: 与之对齐的标准答案列表。

            Returns:
                list[float]: 与 completions 等长的奖励列表。
            """
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
        """只存 LoRA 权重，基座不动，几十兆就能带走这次训练的全部成果。

        Args:
            run_dir: 本次 run 的目录，adapter 落在它下面的 adapter/。
        """

        self.model.save_lora(str(run_dir / "adapter"))

    def build_messages(self, image, problem: str) -> list[dict]:
        """拼评测时的 chat 消息，问法与训练时逐字一致。

        Args:
            image: 待数数的图像。
            problem: 那句问题。

        Returns:
            list[dict]: 一条 chat 格式的消息。
        """

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
        """让模型回答一道数数题。

        评测走确定性生成（`do_sample=False`），训练前后的结果才可以直接比。

        Args:
            image: 待数数的图像。
            problem: 那句问题。
            max_new_tokens: 生成长度上限。

        Returns:
            str: 模型生成的那段回答，已去掉 prompt 部分。
        """

        messages = self.build_messages(image, problem)
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
        """建好 run 目录，并按阶段决定加载可训练模型还是评测模型。

        Args:
            stage: Lightning 传入的阶段名（fit / predict / test）。
        """

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
        """把训练整个交给 GRPOTrainer，所以这一步只会被执行一次。

        Args:
            batch: dataloader 递来的样本，这里用不上——组采样在 GRPOTrainer 内部完成。
            batch_idx: 批次序号。

        Returns:
            torch.Tensor: 一个零标量，只为满足 Lightning 对 training_step 返回值的约定。
        """

        if not self.has_trained:
            self.grpo_trainer.train()
            self.vlm.save_adapter(self.run_dir)
            self.has_trained = True
        return torch.zeros((), device=self.device)

    def predict_step(self, batch, batch_idx):
        """逐条生成回答并累积，供 epoch 结束时统一落盘。

        Args:
            batch: 一批评测样本。
            batch_idx: 批次序号。

        Returns:
            list[dict]: 这一批的预测结果。
        """

        rows = [self.predict_one(example) for example in batch]
        self.prediction_rows.extend(rows)
        return rows

    def on_predict_epoch_end(self) -> None:
        """把逐题预测写成 JSONL，答错的题事后能一条条翻出来看。"""

        output_jsonl = self.run_dir / "eval" / f"{self.prediction_name}_predictions.jsonl"
        output_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in self.prediction_rows) + "\n")

    def test_step(self, batch, batch_idx):
        """逐条生成并打分。

        Args:
            batch: 一批评测样本。
            batch_idx: 批次序号。

        Returns:
            dict[str, float]: 这一批的准确率与格式合规率。
        """

        rows = [self.predict_one(example) for example in batch]
        self.test_rows.extend(rows)
        return score_rows(rows)

    def on_test_epoch_end(self) -> None:
        """把整套题的汇总写成 JSON，这份文件就是讲义里前后对照表的出处。"""

        summary_json = self.run_dir / "eval" / f"{self.prediction_name}_summary.json"
        summary_json.write_text(json.dumps(score_rows(self.test_rows), ensure_ascii=False, indent=2) + "\n")

    def predict_one(self, example: dict) -> dict:
        """回答一道题并记下题面、标准答案和模型输出。

        Args:
            example: 一条评测样本。

        Returns:
            dict: 含 index、problem、target、completion、prediction_name。
        """

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
        """返回 None：优化器由 GRPOTrainer 自己管，Lightning 这边不需要再建一个。

        Returns:
            None
        """

        return None


def score_rows(rows: list[dict]) -> dict[str, float]:
    """用与训练时同一对奖励函数给评测结果打分。

    复用奖励函数不是偷懒——训练看的和评测看的必须是同一把尺子。

    Args:
        rows: predict_one 产出的结果行。

    Returns:
        dict[str, float]: 题数、数数准确率、答案格式合规率。
    """

    if not rows:
        return {"count": 0, "accuracy": 0.0, "format": 0.0}

    accuracy = sum(correctness_reward(row["completion"], row["target"]) for row in rows) / len(rows)
    format_score = sum(format_reward(row["completion"]) for row in rows) / len(rows)
    return {"count": len(rows), "accuracy": accuracy, "format": format_score}


def main() -> None:
    """改 running_stage 切换三个阶段：fit 训练、predict 写 JSONL、test 写 summary。"""

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
