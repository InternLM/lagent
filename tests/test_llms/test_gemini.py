from google import genai
from google.genai import types
from google.genai.types import HttpOptions

client = genai.Client(api_key=" ", base_url='http://100.100.72.89:3888')
model_name = "gemini-2.0-flash-thinking-exp-01-21" # 确保使用支持思考的模型

print("=== 第一轮请求 (Gemini) ===")
# 使用 Stateful 的 Chats API 可以最直观地体现上下文复用
chat = client.chats.create(model=model_name)

response1 = chat.send_message("解释一下量子纠缠，要先自己仔细思考一遍。")

# 【核心验证点】：遍历 Parts，查看 Google 是如何把思考和内容分开的
for part in response1.candidates[0].content.parts:
    # 思考块在 SDK 层会有额外的标识 (取决于具体版本，有时是以纯文本加上特定标记返回，目前最新版有单独处理)
    if getattr(part, 'thought', False): # 或者检查是否有 thought 属性
        print(f"🧠 [思考过程 (Thought/Summary)]: {part.text[:50]}...")
    else:
        print(f"💬 [最终回答]: {part.text[:50]}...\n")

print("=== 第二轮请求（Chat对象自动回传上下文） ===")
# 当我们调用 send_message 时，SDK 底层抓取了 response1 的全套 Parts (包含思考和加密状态) 发回。
response2 = chat.send_message("能不能用通俗的比喻把你刚才想的过程讲一遍？")

print("✅ 第二轮请求成功！")
print(f"💬 [第二轮回答]: {response2.text[:50]}...")
print("💡 结论：Gemini 把复用机制封装在了 Chat Session 里，内部自动携带了第一轮的 thought parts，防止开发者手动拼接出错或篡改。")