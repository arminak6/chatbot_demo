from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re

AWS_REGION = "eu-central-1"
MODEL_ID = "openai.gpt-oss-20b-1:0"

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a professional AI assistant.\n"
        "\n"
        "Rules:\n"
        "- Provide ONLY the final answer to the user.\n"
        "- Do NOT explain your reasoning or thought process.\n"
        "- Do NOT show intermediate steps or analysis.\n"
        "- Keep answers short, clear, and simple.\n"
        "- If you do not know the answer, say \"I don't know\".\n"
        "- Never invent facts.\n"
    )
)

def clean_response(text):
    cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def create_llm():
    return ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

def main():
    llm = create_llm()
    history = [SYSTEM_PROMPT]

    print("LangChain + Bedrock chatbot started. Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        history.append(HumanMessage(content=user_text))

        response = llm.invoke(history)

        cleaned_content = clean_response(response.content)
        print(f"\nBot: {cleaned_content}\n")

        history.append(AIMessage(content=response.content))

if __name__ == "__main__":
    main()
