import asyncio
import logging
import os
import signal
import sys

# 将 Lagent 项目根目录插入 SYS PATH，确保支持正确引用
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lagent.apps.app import InterclawApp
from lagent.apps.channels.cli import CLIChannel
from lagent.apps.channels.feishu import FeishuChannel
from lagent.services.cron import CronService
from lagent.services.heartbeat import HeartbeatService

# 配置工程化日志格式（生产级别格式）
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("interclaw.main")


def main():
    logger.info("Initializing Interclaw Agent OS...")

    from pathlib import Path

    from lagent.actions.filesystem import EditFileAction, ReadFileAction, WriteFileAction
    from lagent.actions.shell import ShellAction
    from lagent.agents import AsyncAgent
    from lagent.agents.aggregator.context import InternClawContextBuilder
    from lagent.agents.internclaw_agent import AsyncEnvAgent, InternClawAgent, get_tool_prompt
    from lagent.hooks.logger import MessageLogger
    from lagent.llms.model import AsyncAPIClient, ModelConfig, SampleParameters

    model_name = "/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns1_1_mini_official/interns1_1_mini_sft_based_cpt_bs512_epoch1_maxlr3e-5_minlr1e-6_max16k-hf/20260207101512/hf-4374"
    # model_name = "gpt-4o-2024-08-06"
    api_base = "http://10.102.218.28:23333/v1/"
    # api_base = f"http://35.220.164.252:3888/v1beta/models/{model_name}:generateContent"
    api_key = ""
    extra_body = {'enable_thinking': True, 'spaces_between_special_tokens': False}
    proxies = None

    model = AsyncAPIClient(
        model=ModelConfig(model=model_name, base_url=api_base, api_key=api_key, proxy=proxies),
        sample_params=SampleParameters(temperature=0.7, top_p=1.0, top_k=50),
        timeout=600,
        max_retry=5,
        sleep_interval=5,
        extra_body=extra_body,
    )
    workspace = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent/workspace"
    actions = [
        ReadFileAction(workspace=workspace),
        WriteFileAction(workspace=workspace),
        EditFileAction(workspace=workspace),
        ShellAction(working_dir=workspace),
    ]
    aggregator = InternClawContextBuilder(Path(workspace), tools=get_tool_prompt(actions))
    policy = AsyncAgent(llm=model, aggregator=aggregator, hooks=[MessageLogger()])
    env = AsyncEnvAgent(actions=actions)
    agent = InternClawAgent(policy_agent=policy, env_agent=env)

    # 2. 创立 App 容器
    app = InterclawApp(agent_engine=agent)

    # 3. 动态注册外部通道 (External Channels)
    # app自动负责把总线 bus 依赖注入给这些组件
    app.register_channel(
        FeishuChannel,
        app_id=os.getenv("FEISHU_APP_ID", "cli_a92538846ff99cd2"),
        app_secret=os.getenv("FEISHU_APP_SECRET", "EpDAW3TCnqUpyUltr1Q5WfR27j0vX13F"),
        encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY", ""),
        verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN", ""),
    )

    # 如果有飞书 credentials，可动态注册飞书 channel
    # app.register_channel(FeishuChannel, app_id="TODO_APP_ID", app_secret="TODO_APP_SECRET")

    # 4. 动态注册后台服务 (Services & Daemons)
    app.register_service(CronService)
    app.register_service(HeartbeatService, pulse_seconds=10)

    # 5. 启动总控制器
    app.run()


if __name__ == "__main__":
    main()
