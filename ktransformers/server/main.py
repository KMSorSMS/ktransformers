import os
import re
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn.logging
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from ktransformers.server.config.config import Config
from ktransformers.server.utils.create_interface import create_interface
from ktransformers.server.backend.args import default_args
from fastapi.openapi.utils import get_openapi

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from ktransformers.server.api import router, post_db_creation_operations
from ktransformers.server.utils.sql_utils import Base, SQLUtil
from ktransformers.server.config.log import logger


def mount_app_routes(mount_app: FastAPI):
    sql_util = SQLUtil()
    logger.info("Creating SQL tables")
    Base.metadata.create_all(bind=sql_util.sqlalchemy_engine)
    post_db_creation_operations()
    mount_app.include_router(router)


def create_app():
    cfg = Config()
    app = FastAPI()
    if Config().web_cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app)
    if cfg.mount_web:
        mount_index_routes(app)
    return app


def update_web_port(config_file: str):
    ip_port_pattern = (
        r"(localhost|((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)):[0-9]{1,5}"
    )
    with open(config_file, "r", encoding="utf-8") as f_cfg:
        web_config = f_cfg.read()
    ip_port = "localhost:" + str(Config().server_port)
    new_web_config = re.sub(ip_port_pattern, ip_port, web_config)
    with open(config_file, "w", encoding="utf-8") as f_cfg:
        f_cfg.write(new_web_config)


def mount_index_routes(app: FastAPI):
    project_dir = os.path.dirname(os.path.dirname(__file__))
    web_dir = os.path.join(project_dir, "website/dist")
    web_config_file = os.path.join(web_dir, "config.js")
    update_web_port(web_config_file)
    if os.path.exists(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir), name="static")
    else:
        err_str = f"No website resources in {web_dir}, please complile the website by npm first"
        logger.error(err_str)
        print(err_str)
        exit(1)


def run_api(app, host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port, log_level="debug")


def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ktransformers server",
        version="1.0.0",
        summary="This is a server that provides a RESTful API for ktransformers.",
        description="We provided chat completion and openai assistant interfaces.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {"url": "https://kvcache.ai/media/icon_1.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema

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
        parser.add_argument("--device", type=str, default=self.cfg.model_device, help="Warning: Abandoning this parameter")
        parser.add_argument("--gguf_path", type=str, default=self.cfg.gguf_path)
        parser.add_argument("--optimize_config_path", default=None, type=str, required=False)
        parser.add_argument("--cpu_infer", type=int, default=self.cfg.cpu_infer)
        parser.add_argument("--type", type=str, default=self.cfg.backend_type)

        # model configs
        parser.add_argument("--model_cache_lens", type=int, default=self.cfg.model_cache_lens) # int?
        parser.add_argument("--paged", type=bool, default=self.cfg.paged)
        parser.add_argument("--total_context", type=int, default=self.cfg.total_context)
        parser.add_argument("--max_batch_size", type=int, default=self.cfg.max_batch_size)
        parser.add_argument("--max_chunk_size", type=int, default=self.cfg.max_chunk_size)
        parser.add_argument("--max_new_tokens", type=int, default=self.cfg.max_new_tokens)
        parser.add_argument("--json_mode", type=bool, default=self.cfg.json_mode)
        parser.add_argument("--healing", type=bool, default=self.cfg.healing)
        parser.add_argument("--ban_strings", type=list, default=self.cfg.ban_strings, required=False)
        parser.add_argument("--gpu_split", type=str, default=self.cfg.gpu_split, required=False)
        parser.add_argument("--length",type=int, default=self.cfg.length, required=False)
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
        
        return parser.parse_args()

def main():
    cfg = Config()
    arg_parser = ArgumentParser(cfg)

    # 初始化消息
    args = arg_parser.parse_args()
    cfg.model_name = args.model_name
    cfg.model_path = args.model_path
    cfg.model_device = args.device
    cfg.mount_web = args.web
    cfg.server_ip = args.host
    cfg.server_port = args.port
    cfg.cpu_infer = args.cpu_infer
    cfg.backend_type = args.type

    default_args.model_dir = args.model_path
    default_args.device = args.device
    default_args.gguf_path = args.gguf_path
    default_args.optimize_config_path = args.optimize_config_path

    app = create_app()
    custom_openapi(app)
    create_interface(config=cfg, default_args=default_args)
    run_api(
        app=app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
