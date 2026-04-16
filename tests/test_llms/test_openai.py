from openai import OpenAI

client = OpenAI(api_key=" ", base_url='http://100.100.72.89:3888/v1')
model_name = "o3-mini"

print("=== 第一轮请求 (OpenAI o3-mini) ===")
# 使用标准 API 进行测试
messages =[{"role": "user", "content": "用复杂逻辑分析为什么天空是蓝色的。"}]

response1 = client.chat.completions.create(
    model=model_name,
    reasoning_effort="high", # 控制思考深度
    messages=messages
)

assistant_message = response1.choices[0].message
# 【核心验证点 1】：content 里直接就是最终答案，没有任何思考过程
print(f"💬[最终回答]: {assistant_message.content[:50]}...")

# 【核心验证点 2】：真实的思考 Token 数量被藏在 usage 里统计（你得付钱，但看不到内容）
reasoning_tokens = response1.usage.completion_tokens_details.reasoning_tokens
print(f"🧠[隐藏的思考Token数量]: {reasoning_tokens}\n")

print("=== 第二轮请求（依靠会话历史维持逻辑） ===")
# 把它返回的干净 message (没有任何隐藏思考块) 追加回去
messages.append(assistant_message)
messages.append({"role": "user", "content": "好的，根据你刚才的逻辑，解释一下晚霞为什么是红的。"})

response2 = client.chat.completions.create(
    model=model_name,
    reasoning_effort="high",
    messages=messages
)
print("✅ 第二轮请求成功。虽然你看不到思考过程，但模型通过上一轮生成的最终回答重建了逻辑链。")

# 备注：在 Assistants API 中，你甚至不需要传 messages 数组，
# 只需要传入 thread_id: client.beta.threads.messages.create(thread_id="xxx", ...)
# 服务端会自动提取隐藏在该 Thread 里的加密思考上下文（Encrypted Reasoning Items）。