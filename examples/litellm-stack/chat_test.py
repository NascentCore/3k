import openai

# 配置 litellm 的端点和 API 密钥
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://{node_ip}:31000/v1"
)

def call_model(model_name, prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100
        )
        print(f"Model: {model_name}\nQuestion:{prompt}\nResponse: {response.choices[0].message.content}\n")
    except Exception as e:
        print(f"Error calling model {model_name}: {e}")

# 调用 Llama 模型
call_model("llama-31-8b-instruct", "What is the capital of France?")

# 调用 Mistral 模型
call_model("mistral-7b-instruct", "Explain quantum computing in simple terms.")
