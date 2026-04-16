import anthropic

client = anthropic.Anthropic(api_key=" ", base_url='http://100.100.72.89:3888')
model_name = "claude-sonnet-4-20250514-thinking"

# 第一轮请求：开启 Extended Thinking
print("=== 第一轮请求 (Claude) ===")
response1 = client.messages.create(
    model=model_name,
    max_tokens=4096,
    thinking={
        "type": "enabled",
        "budget_tokens": 1024 # 给 1024 个 token 的思考预算
    },
    messages=[{"role": "user", "content": "请用极其复杂的逻辑证明 1+1=2，必须深入思考。"}]
)

# 打印解析结果
for block in response1.content:
    if block.type == "thinking":
        print(f"🧠 [思考过程]: {block.thinking[:50]}...")
        print(f"🔒[防篡改签名]: {block.signature[:30]}...\n")
    elif block.type == "text":
        print(f"💬 [最终回答]: {block.text[:50]}...\n")

# 第二轮请求：状态复用（必须原封不动传回完整 Content）
print("=== 第二轮请求（复用思考上下文） ===")
messages_history =[
    {"role": "user", "content": "请用极其复杂的逻辑证明 1+1=2，必须深入思考。"},
    # 【核心点】直接传入上一次的 response1.content (包含 text 和 thinking 及其 signature)
    {"role": "assistant", "content": response1.content}, 
    {"role": "user", "content": "基于你刚才的思考，再推导一下 1+2=3。"}
]

try:
    response2 = client.messages.create(
        model=model_name,
        max_tokens=4096,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=messages_history
    )
    print("✅ 第二轮请求成功！Claude 成功验证了上一次的思考签名。")
    
    # 【破坏性试验】如果你尝试修改 response1.content 里 thinking 的任何一个字，
    # 再次请求时，Anthropic API 会直接抛出 400 错误：Signature Validation Failed.
except Exception as e:
    print(f"❌ 请求失败: {e}")