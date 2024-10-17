import argparse
from ktransformers.server.backend.args import ConfigArgs, default_args


class ArgumentParser:
    def __init__(self, cfg):
        self.cfg = cfg

    def parse_args(self):
        parser = argparse.ArgumentParser(prog="kvcache.ai", description="Ktransformers")
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=self.cfg.server_port)
        parser.add_argument("--ssl_keyfile", type=str)
        parser.add_argument("--ssl_certfile", type=str)
        parser.add_argument("--web", type=bool, default=False)
        parser.add_argument("--model_name", type=str, default=self.cfg.model_name)
        parser.add_argument("--model_path", type=str, default=self.cfg.model_path)
        parser.add_argument(
            "--device", type=str, default=self.cfg.model_device, help="Warning: Abandoning this parameter"
        )
        parser.add_argument("--gguf_path", type=str, default=self.cfg.gguf_path)
        parser.add_argument("--optimize_config_path", default=None, type=str, required=False)
        parser.add_argument("--cpu_infer", type=int, default=self.cfg.cpu_infer)
        parser.add_argument("--type", type=str, default=self.cfg.backend_type)

        # model configs
        parser.add_argument("--model_cache_lens", type=int, default=self.cfg.model_cache_lens)  # int?
        parser.add_argument("--paged", type=bool, default=self.cfg.paged)
        parser.add_argument("--total_context", type=int, default=self.cfg.total_context)
        parser.add_argument("--max_batch_size", type=int, default=self.cfg.max_batch_size)
        parser.add_argument("--max_chunk_size", type=int, default=self.cfg.max_chunk_size)
        parser.add_argument("--max_new_tokens", type=int, default=self.cfg.max_new_tokens)
        parser.add_argument("--json_mode", type=bool, default=self.cfg.json_mode)
        parser.add_argument("--healing", type=bool, default=self.cfg.healing)
        parser.add_argument("--ban_strings", type=list, default=self.cfg.ban_strings, required=False)
        parser.add_argument("--gpu_split", type=str, default=self.cfg.gpu_split, required=False)
        parser.add_argument("--length", type=int, default=self.cfg.length, required=False)
        parser.add_argument("--rope_scale", type=float, default=self.cfg.rope_scale, required=False)
        parser.add_argument("--rope_alpha", type=float, default=self.cfg.rope_alpha, required=False)
        parser.add_argument("--no_flash_attn", type=bool, default=self.cfg.no_flash_attn)
        parser.add_argument("--low_mem", type=bool, default=self.cfg.low_mem)
        parser.add_argument("--experts_per_token", type=int, default=self.cfg.experts_per_token, required=False)
        parser.add_argument("--load_q4", type=bool, default=self.cfg.load_q4)
        parser.add_argument("--fast_safetensors", type=bool, default=self.cfg.fast_safetensors)
        parser.add_argument("--draft_model_dir", type=str, default=self.cfg.draft_model_dir, required=False)
        parser.add_argument("--no_draft_scale", type=bool, default=self.cfg.no_draft_scale)
        parser.add_argument("--modes", type=bool, default=self.cfg.modes)
        parser.add_argument("--mode", type=str, default=self.cfg.mode)
        parser.add_argument("--username", type=str, default=self.cfg.username)
        parser.add_argument("--botname", type=str, default=self.cfg.botname)
        parser.add_argument("--system_prompt", type=str, default=self.cfg.system_prompt, required=False)
        parser.add_argument("--temperature", type=float, default=self.cfg.temperature)
        parser.add_argument("--smoothing_factor", type=float, default=self.cfg.smoothing_factor)
        parser.add_argument("--dynamic_temperature", type=str, default=self.cfg.dynamic_temperature, required=False)
        parser.add_argument("--top_k", type=int, default=self.cfg.top_k)
        parser.add_argument("--top_p", type=float, default=self.cfg.top_p)
        parser.add_argument("--top_a", type=float, default=self.cfg.top_a)
        parser.add_argument("--skew", type=float, default=self.cfg.skew)
        parser.add_argument("--typical", type=float, default=self.cfg.typical)
        parser.add_argument("--repetition_penalty", type=float, default=self.cfg.repetition_penalty)
        parser.add_argument("--frequency_penalty", type=float, default=self.cfg.frequency_penalty)
        parser.add_argument("--presence_penalty", type=float, default=self.cfg.presence_penalty)
        parser.add_argument("--max_response_tokens", type=int, default=self.cfg.max_response_tokens)
        parser.add_argument("--response_chunk", type=int, default=self.cfg.response_chunk)
        parser.add_argument("--no_code_formatting", type=bool, default=self.cfg.no_code_formatting)
        parser.add_argument("--cache_8bit", type=bool, default=self.cfg.cache_8bit)
        parser.add_argument("--cache_q4", type=bool, default=self.cfg.cache_q4)
        parser.add_argument("--ngram_decoding", type=bool, default=self.cfg.ngram_decoding)
        parser.add_argument("--print_timings", type=bool, default=self.cfg.print_timings)
        parser.add_argument("--amnesia", type=bool, default=self.cfg.amnesia)
        parser.add_argument("--batch_size", type=int, default=self.cfg.batch_size)
        parser.add_argument("--cache_lens", type=int, default=self.cfg.cache_lens)

        # log configs
        # log level: debug, info, warn, error, crit
        parser.add_argument("--log_dir", type=str, default=self.cfg.log_dir)
        parser.add_argument("--log_file", type=str, default=self.cfg.log_file)
        parser.add_argument("--log_level", type=str, default=self.cfg.log_level)
        parser.add_argument("--backup_count", type=int, default=self.cfg.backup_count)

        # db configs
        parser.add_argument("--db_type", type=str, default=self.cfg.db_type)
        parser.add_argument("--db_host", type=str, default=self.cfg.db_host)
        parser.add_argument("--db_port", type=str, default=self.cfg.db_port)
        parser.add_argument("--db_name", type=str, default=self.cfg.db_name)
        parser.add_argument("--db_pool_size", type=int, default=self.cfg.db_pool_size)
        parser.add_argument("--db_database", type=str, default=self.cfg.db_database)

        # user config
        parser.add_argument("--user_secret_key", type=str, default=self.cfg.user_secret_key)
        parser.add_argument("--user_algorithm", type=str, default=self.cfg.user_algorithm)

        # web config
        parser.add_argument("--web_cross_domain", type=bool, default=self.cfg.web_cross_domain)

        # file config
        parser.add_argument("--file_upload_dir", type=str, default=self.cfg.file_upload_dir)
        parser.add_argument("--assistant_store_dir", type=str, default=self.cfg.assistant_store_dir)

        args = parser.parse_args()
        # set config from args
        for key, value in vars(args).items():
            if value is not None and hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
        # we add the name not match args individually
        self.cfg.model_device = args.device
        self.cfg.mount_web = args.web
        self.cfg.server_ip = args.host
        self.cfg.server_port = args.port
        self.cfg.backend_type = args.type
        cfg = self.cfg

        default_args.model_name = cfg.model_name
        default_args.model_dir = cfg.model_path
        default_args.optimize_config_path = cfg.optimize_config_path
        default_args.gguf_path = cfg.gguf_path
        default_args.paged = cfg.paged
        default_args.total_context = cfg.total_context
        default_args.max_batch_size = cfg.max_batch_size
        default_args.max_chunk_size = cfg.max_chunk_size
        default_args.max_new_tokens = cfg.max_new_tokens
        default_args.json_mode = cfg.json_mode
        default_args.healing = cfg.healing
        default_args.ban_strings = cfg.ban_strings
        default_args.gpu_split = cfg.gpu_split
        default_args.length = cfg.length
        default_args.rope_scale = cfg.rope_scale
        default_args.rope_alpha = cfg.rope_alpha
        default_args.no_flash_attn = cfg.no_flash_attn
        default_args.low_mem = cfg.low_mem
        default_args.experts_per_token = cfg.experts_per_token
        default_args.load_q4 = cfg.load_q4
        default_args.fast_safetensors = cfg.fast_safetensors
        default_args.draft_model_dir = cfg.draft_model_dir
        default_args.no_draft_scale = cfg.no_draft_scale
        default_args.modes = cfg.modes
        default_args.mode = cfg.mode
        default_args.username = cfg.username
        default_args.botname = cfg.botname
        default_args.system_prompt = cfg.system_prompt
        default_args.temperature = cfg.temperature
        default_args.smoothing_factor = cfg.smoothing_factor
        default_args.dynamic_temperature = cfg.dynamic_temperature
        default_args.top_k = cfg.top_k
        default_args.top_p = cfg.top_p
        default_args.top_a = cfg.top_a
        default_args.skew = cfg.skew
        default_args.typical = cfg.typical
        default_args.repetition_penalty = cfg.repetition_penalty
        default_args.frequency_penalty = cfg.frequency_penalty
        default_args.presence_penalty = cfg.presence_penalty
        default_args.max_response_tokens = cfg.max_response_tokens
        default_args.response_chunk = cfg.response_chunk
        default_args.no_code_formatting = cfg.no_code_formatting
        default_args.cache_8bit = cfg.cache_8bit
        default_args.cache_q4 = cfg.cache_q4
        default_args.ngram_decoding = cfg.ngram_decoding
        default_args.print_timings = cfg.print_timings
        default_args.amnesia = cfg.amnesia
        default_args.batch_size = cfg.batch_size
        default_args.cache_lens = cfg.cache_lens
        default_args.device = cfg.device

        return args
