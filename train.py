"""
copy from: /Open-Llama/train_lm.py
"""
import yaml
import math
import logging
from absl import app
from absl import flags
from accelerate import Accelerator
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from datasets.distributed import split_dataset_by_node
from models.tokenization_vocab_32k_gpt2 import vocab_32k_gpt2Tokenizer
#from transformers import AutoConfig, AutoModelForCausalLM
from models.configuration_vocab_32k_gpt2_moe import vocab_32k_gpt2moeConfig
from models.modeling_vocab_32k_gpt2_moe import vocab_32k_GPT2MOELMHeadModel
from dataset.dataset import construct_dataset
from trainer import Trainer
import os 


FLAGS = flags.FLAGS
flags.DEFINE_string("train_config", None, "Training config path")
flags.DEFINE_string("model_config", None, "Model config path")


def main(argv):
    with open(FLAGS.train_config, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    data_config = config["data"]
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"].get("gradient_accumulation_steps", 1)
    )
    tokenizer = vocab_32k_gpt2Tokenizer(vocab_file=data_config["tokenizer_model_path"], legacy=False)
    if data_config.get("split_by_shard", False):
        train_dataset = construct_dataset(
            data_config, tokenizer, world_size=accelerator.num_processes
        )
    else:
        train_dataset = construct_dataset(data_config, tokenizer)
    train_dataset = split_dataset_by_node(
        train_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["train_batch_size"],
        num_workers=config["train"]["train_num_workers_4_dataloader"],
        prefetch_factor=config["train"].get("prefetch_factor", 2),
        pin_memory=True,
    )
    vocab_size = tokenizer.vocab_size

    model_config = vocab_32k_gpt2moeConfig.from_pretrained(FLAGS.model_config)
    model_config.vocab_size = vocab_size
    model_config.pad_token_id = tokenizer.pad_id
    # 使用AutoModel可以在Deepspeed.zero.Init()下正确的生效，而直接使用如OpenLlamaModel不能正确生效，导致浪费大量内存空间
    # https://github.com/huggingface/accelerate/pull/932
    if config["train"]["ckpt"] is not None:
        raw_model = vocab_32k_GPT2MOELMHeadModel.from_pretrained(
            config["train"]["ckpt"], config=model_config
        )
        logging.warning("Loaded ckpt from: {}".format(config["train"]["ckpt"]))
    else:
        raw_model = vocab_32k_GPT2MOELMHeadModel(config=model_config)
    
    total_params = sum(param.numel() for param in raw_model.parameters())
    logging.warning("#parameters: {}".format(total_params))
    # lora
    if config["train"].get("use_lora", False):
        # gradient ckpt bug, https://github.com/huggingface/transformers/issues/23170
        if hasattr(raw_model, "enable_input_require_grads"):
            raw_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            raw_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        raw_model = get_peft_model(raw_model, peft_config)
        raw_model.print_trainable_parameters()
    if config["train"].get("gradient_checkpointing_enable", False):
        raw_model.gradient_checkpointing_enable()
    trainer = Trainer(config, raw_model, train_loader, tokenizer, accelerator)

    # for batch in train_loader:
    #     print(batch)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
