from openai import OpenAI
from config import OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def llm(query, system_prompt=None, history=[], user_stop_words=[], temperature=0.7):
    """LLM function - Use OpenAI API with configurable system prompt"""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant specialized in flight booking assistance. Help users find the best flights based on their needs."
    
    messages = [{"role": "system", "content": system_prompt}]
    for hist in history:
        messages.append({"role": "user", "content": hist[0]})
        messages.append({"role": "assistant", "content": hist[1]})
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            stop=user_stop_words
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)