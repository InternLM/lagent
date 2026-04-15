import asyncio
import json
import logging
import threading
from typing import Any

from lagent.apps.bus import MessageBus, InboundEvent
from lagent.apps.channels.base import BaseChannel

logger = logging.getLogger("lagent.interclaw.channels.feishu")

class FeishuChannel(BaseChannel):
    """
    基于 lark-oapi 实现的飞书全双工通道。
    内部维护一个独立的 WebSocket 线程监听飞书事件，用 async loop 发送回调回复。
    """
    def __init__(self, bus: MessageBus, app_id: str, app_secret: str, encrypt_key: str = "", verification_token: str = ""):
        super().__init__(bus, channel_name="feishu")
        self.app_id = app_id
        self.app_secret = app_secret
        self.encrypt_key = encrypt_key
        self.verification_token = verification_token
        
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _producer_loop(self):
        """Producer: 将飞书的回调变成 InboundEvent (使用 Lark WS 客户端挂载后台线程)"""
        try:
            import lark_oapi as lark
        except ImportError:
            logger.error("lark-oapi sdk is missing! Please pip install lark-oapi")
            return

        self._loop = asyncio.get_running_loop()

        # 1. 构造主动发信 Client
        self._client = lark.Client.builder() \
            .app_id(self.app_id) \
            .app_secret(self.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()

        # 2. 构造事件处理器 (收信器)
        def _on_message_callback(data: Any) -> None:
            # 这个回调是在飞书内置的多线程里触发的，所以必须丢回 asyncio 中枢
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(self._handle_feishu_inbound(data), self._loop)

        event_handler = lark.EventDispatcherHandler.builder(
            self.encrypt_key,
            self.verification_token,
        ).register_p2_im_message_receive_v1(
            _on_message_callback
        ).build()

        # 3. 构造长连接 Web Socket 客户端
        self._ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )

        def run_ws():
            logger.info("Feishu WebSocket is starting via Lark-OAPI...")
            
            # 1. 为当前的后台线程创建一个独立的、全新的事件循环
            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            
            # 2. 核心修复：强制覆盖 lark_oapi.ws.client 模块内部缓存的 loop 变量
            # 防止其去调度主线程的 loop 而引发 "This event loop is already running" 错误
            import lark_oapi.ws.client
            lark_oapi.ws.client.loop = ws_loop

            try:
                self._ws_client.start()
            except Exception as e:
                logger.error(f"Feishu WebSocket failed: {e}")

        # 挂载到一个守护线程执行
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()

        # 主协程挂起，维持存活
        while True:
            await asyncio.sleep(3600)

    async def _handle_feishu_inbound(self, data: Any):
        """将飞书原生的数据格式转化成 Agent OS 的内部格式并丢入总线"""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            if sender.sender_type == "bot":
                return # 不自己跟自己聊天

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            msg_type = message.message_type

            # 构建会话 ID: 例如 feishu:group_xabc123 或者 feishu:user_888
            session_id = f"feishu:{chat_id}"

            # 解析内容 (简化版文本提取，实际你需要保留 nanobot 里的全面解析如图片、Card等)
            content = ""
            if msg_type == "text":
                content_json = json.loads(message.content) if message.content else {}
                content = content_json.get("text", "")
            
            if content.strip():
                logger.info(f"[Feishu Inbound] Receive MSG -> session:{session_id}")
                await self.bus.publish_inbound(InboundEvent(
                    channel=self.channel_name,
                    session_id=session_id,
                    content=content,
                    # 可以通过 Metadata 保存原始的飞书数据对象以备高级工具使用
                    metadata={"feishu_chat_id": chat_id, "feishu_receive_id_type": "chat_id"}
                ))
        except Exception as e:
            logger.error(f"Failed to process Feishu inbound message: {e}")

    def _sync_send_text(self, receive_id: str, text: str):
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
        request = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type("text")
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .build()
            ).build()
        self._client.im.v1.message.create(request)

    async def _consumer_loop(self):
        """Consumer: 监听由 Dispatcher 分发给专属 Feishu Channel 的回复"""
        while True:
            try:
                event = await self.outbound_queue.get()
                logger.info(f"[Feishu Outbound] Send MSG -> {event.session_id}")
                
                # 从 session_id 中解析出飞书认的 chat_id (feishu:chat_xxxxx)
                if ":" in event.session_id:
                    _, chat_id = event.session_id.split(":", 1)
                else:
                    chat_id = event.session_id
                    
                # 非阻塞投递
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._sync_send_text, chat_id, event.content)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Failed to send outbound to Feishu: {e}")

    async def start(self):
        logger.info("Feishu Channel initialized. Connecting pipelines...")
        tasks = [
            asyncio.create_task(self._producer_loop(), name="Feishu_Producer"),
            asyncio.create_task(self._consumer_loop(), name="Feishu_Consumer"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()

if __name__ == "__main__":
    # 这个 main 只是为了本地测试飞书通道的收发功能，实际使用时不需要运行这个文件
    import os
    from lagent.apps.bus import MessageBus
    logging.basicConfig(
        level=logging.INFO,  # 允许打印 INFO 及以上级别的日志 (如果你想看 debug 就改成 logging.DEBUG)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    bus = MessageBus()
    feishu_channel = FeishuChannel(
        bus=bus,
        app_id=os.getenv("FEISHU_APP_ID", "cli_a92538846ff99cd2"),
        app_secret=os.getenv("FEISHU_APP_SECRET", "EpDAW3TCnqUpyUltr1Q5WfR27j0vX13F"),
        encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY", ""),
        verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN", "")
    )

    asyncio.run(feishu_channel.start())