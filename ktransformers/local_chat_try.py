import os
import platform
import sys
from uuid import uuid4
from ktransformers.server.config.config import Config
from ktransformers.server.backend.args import default_args
from ktransformers.server.utils.create_interface import create_interface
from ktransformers.server.backend.base import BackendInterfaceBase
from ktransformers.server.utils.create_interface import get_interface

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate
from ktransformers.server.config.config import Config

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}


def local_chat(
    model_path: str | None = None,
    optimize_rule_path: str = None,
    gguf_path: str | None = None,
    max_new_tokens: int = 1000,
    cpu_infer: int = Config().cpu_infer,
    use_cuda_graph: bool = True,
    prompt_file: str | None = None,
    mode: str = "normal",
):

    torch.set_grad_enabled(False)
    cfg = Config()
    cfg.backend_type = "ktransformers":
    cfg.cpu_infer = cpu_infer
    cfg.model_path = model_path
    cfg.mode = mode
    cfg.max_new_tokens = max_new_tokens

    default_args.max_new_tokens = cfg.max_new_tokens
    default_args.model_dir = cfg.model_path

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if optimize_rule_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_rule_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_rule_path = input("please input the path of your rule file(yaml file containing optimize rules):")

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to"
            " current model):"
        )
    create_interface(config=cfg, default_args=default_args)


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # if mode == "long_context":
    #     assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
    #     torch.set_default_dtype(torch.float16)
    # else:
    #     torch.set_default_dtype(config.torch_dtype)

    # with torch.device("meta"):
    #     if config.architectures[0] in custom_models:
    #         print("using custom modeling_xxx.py.")
    #         if "Qwen2Moe" in config.architectures[0]:  # Qwen2Moe must use flash_attention_2 to avoid overflow.
    #             config._attn_implementation = "flash_attention_2"
    #         if "Llama" in config.architectures[0]:
    #             config._attn_implementation = "eager"
    #         if "Mixtral" in config.architectures[0]:
    #             config._attn_implementation = "flash_attention_2"

    #         model = custom_models[config.architectures[0]](config)
    #     else:
    #         model = AutoModelForCausalLM.from_config(
    #             config, trust_remote_code=True, attn_implementation="flash_attention_2"
    #         )

    # optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)

    # model.generation_config = GenerationConfig.from_pretrained(model_path)
    # if model.generation_config.pad_token_id is None:
    #     model.generation_config.pad_token_id = model.generation_config.eos_token_id
    # model.eval()
    logging.basicConfig(level=logging.INFO)

    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    # add a history chat content
    his_content = []
    interface: BackendInterfaceBase = get_interface()
    while True:
        content = input("Chat: ")
        if content.startswith('"""'):  # prefix """
            # multi lines input
            content = content[3:] + "\n"
            while True:
                line = input("")
                if line.endswith('"""'):
                    # end multi lines input
                    line = line[:-3]  # suffix """
                    if line:
                        content += line + "\n"
                    break
                else:
                    content += line + "\n"

        if content == "":
            if prompt_file != None:
                content = open(prompt_file, "r").read()
            else:
                content = "Please write a piece of quicksort code in C++."
        elif os.path.isfile(content):
            content = open(content, "r").read()
        messages = his_content + [{"role": "user", "content": content}]
        print("messages:", messages)
        # input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if mode == "long_context":
            assert (
                Config().long_context_config["max_seq_len"] > input_tensor.shape[1] + max_new_tokens
            ), "please change max_seq_len in  ~/.ktransformers/config.yaml"
        torch.set_default_dtype(torch.bfloat16)  # TODO: Remove this, replace dtype using config

        """
        async def inference(self, local_messages, thread_id: str):
            self.profiler.create_and_start_timer("tokenize")
            if isinstance(local_messages, List):
                input_ids = self.format_and_tokenize_input_ids(thread_id, local_messages)
            elif isinstance(local_messages, str):
                input_ids = self.tokenize_prompt(local_messages)
        """
        # generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode)
        generated=""
        id = str(uuid4())

        # TODO: messages' format???
        async for token in interface.inference(messages, id):
            generated+=token
        his_content += [
            {"role": "user", "content": content},
            {"role": "assitant", "content": generated},
        ]


if __name__ == "__main__":
    fire.Fire(local_chat)