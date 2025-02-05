from lagent.llms import GPTStyleAPI

def chat_xinfrence():
    api_base = 'http://192.168.26.213:13000/v1/chat/completions' # oneapi
    model_name = "deepseek-r1:1.5b"
    gpttool = GPTStyleAPI(
        model_type=model_name,
        api_base=api_base,
        key="sk-IXgCTwuoEwxL1CiBE4744688D8094521B70f4aDeE6830c5e",
        retry=3,
        meta_template=None,
        max_new_tokens=512,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        repetition_penalty=1,
        stream=False,
        stop_words=None,
    )
    res = gpttool.chat(inputs=[
        {
            "role": "user",
            "content": "世界第一高峰是"
        }])
    #     res = gpttool.generate(inputs="世界第一高峰是")
    print(res)
def chat_ollama():
    api_base = 'http://192.168.26.212:11434/api/chat'  # ollama
    model_name = "qwen:7b"
    gpttool = GPTStyleAPI(
        model_type=model_name,
        api_base=api_base,
        key="sk-IXgCTwuoEwxL1CiBE4744688D8094521B70f4aDeE6830c5e",
        retry=3,
        meta_template=None,
        max_new_tokens=512,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        repetition_penalty=1,
        stream=False,
        stop_words=None,
    )
    res = gpttool.chat(inputs=[
        {
            "role": "user",
            "content": "世界第一高峰是"
        }])
    print(res)

def chat_direct():
    api_base = 'http://192.168.26.213/v1/chat/completions'  # 直连
    model_name = "Baichuan2-Turbo"
    gpttool = GPTStyleAPI(
        model_type=model_name,
        api_base=api_base,
        key="sk-IXgCTwuoEwxL1CiBE4744688D8094521B70f4aDeE6830c5e",
        retry=3,
        meta_template=None,
        max_new_tokens=512,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        repetition_penalty=1,
        stream=False,
        stop_words=None,
    )
    res = gpttool.chat(inputs=[
        {
            "role": "user",
            "content": "世界第一高峰是"
        }])
    print(res)

def chat_lmdeploy():
    api_base = 'http://192.168.26.212:24444/v1/chat/completions'  # 直连
    model_name = "deepseek-r1:14b"
    gpttool = GPTStyleAPI(
        model_type=model_name,
        api_base=api_base,
        key="sk-IXgCTwuoEwxL1CiBE4744688D8094521B70f4aDeE6830c5e",
        retry=3,
        meta_template=None,
        max_new_tokens=512,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        repetition_penalty=1,
        stream=False,
        stop_words=None,
    )
    res = gpttool.chat(inputs=[
        {
            "role": "user",
            "content": "世界第一高峰是"
        }])
    print(res)

def chat_oneapi():
    api_base = 'http://192.168.26.213:13000/v1/chat/completions' # oneapi
    model_name = "deepseek-r1-14b"
    gpttool = GPTStyleAPI(
        model_type=model_name,
        api_base=api_base,
        key="sk-CZOUavQGNzkkQjZr626908A0011040F8B743C526F315D6Ee",
        retry=3,
        meta_template=None,
        max_new_tokens=512,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        repetition_penalty=1,
        stream=False,
        stop_words=None,
    )
    res = gpttool.chat(inputs=[
        {
            "role": "user",
            "content": "世界第一高峰是"
        }])
    print(res)

def stream_chat_ollama():
    api_base = 'http://192.168.26.212:11434/api/chat'  # ollama
    model_name = "qwen:7b"
    gpttool = GPTStyleAPI(
        model_type=model_name,
        api_base=api_base,
        key="sk-IXgCTwuoEwxL1CiBE4744688D8094521B70f4aDeE6830c5e",
        retry=3,
        meta_template=None,
        max_new_tokens=512,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        repetition_penalty=1,
        stream=False,
        stop_words=None,
    )
    res = gpttool.stream_chat(inputs=[
        {
            "role": "user",
            "content": "世界第一高峰是"
        }])
    for status, content, _ in res:
        print(content, end='', flush=True)

def stream_chat_oneapi():
    api_base = 'http://192.168.26.213:13000/v1/chat/completions' # oneapi
    model_name = "deepseek-r1-14b"
    gpttool = GPTStyleAPI(
        model_type=model_name,
        api_base=api_base,
        key="sk-CZOUavQGNzkkQjZr626908A0011040F8B743C526F315D6Ee",
        retry=3,
        meta_template=None,
        max_new_tokens=512,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        repetition_penalty=1,
        stream=False,
        stop_words=None,
    )
    res = gpttool.stream_chat(inputs=[
        {
            "role": "user",
            "content": "世界第一高峰是"
        }])
    for status, content, _ in res:
        print(content, end='', flush=True)

if __name__ == '__main__':
    # chat_xinfrence()
    # chat_direct()
    # chat_ollama()
    # chat_oneapi()
    chat_lmdeploy()

    # #流式输出测试
    # stream_chat_ollama()
    # stream_chat_oneapi()