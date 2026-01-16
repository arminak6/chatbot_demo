from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
from opensearch_client import OpenSearchKnowledgeBase
import config

def clean_response(text):
    """Remove reasoning tags and clean response"""
    cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def create_system_prompt(use_context: bool = False, context: str = "") -> SystemMessage:
    """
    Create system prompt based on whether we have knowledge base context
    
    Args:
        use_context: Whether context from knowledge base is available
        context: The formatted context from knowledge base documents
    
    Returns:
        SystemMessage with appropriate instructions
    """
    base_rules = (
        "You are a professional AI assistant.\n"
        "\n"
        "Rules:\n"
        "- Provide ONLY the final answer to the user.\n"
        "- Do NOT explain your reasoning or thought process.\n"
        "- Do NOT show intermediate steps or analysis.\n"
        "- Keep answers short, clear, and simple.\n"
        "- Never invent facts.\n"
    )
    
    if use_context and context:
        prompt_content = (
            f"{base_rules}"
            f"- Answer the question based PRIMARILY on the knowledge base documents provided below.\n"
            f"- Only use your general knowledge if the documents don't contain the answer.\n"
            f"- If the answer is not in the documents and you don't know from general knowledge, say \"I don't know\".\n"
            f"- When answering from the documents, be confident and direct.\n"
            f"\n"
            f"{context}"
        )
    else:
        prompt_content = (
            f"{base_rules}"
            f"- If you do not know the answer, say \"I don't know\".\n"
        )
    
    return SystemMessage(content=prompt_content)


def create_llm():
    """Create and configure the LLM"""
    return ChatBedrock(
        model_id=config.MODEL_ID,
        region_name=config.AWS_REGION,
        model_kwargs={
            "max_tokens": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
        },
    )


def get_answer_with_knowledge_base(
    user_question: str,
    opensearch_kb: OpenSearchKnowledgeBase,
    llm: ChatBedrock,
    conversation_history: list
) -> tuple[str, bool]:
    """
    Get answer by first searching knowledge base, then using LLM
    
    Args:
        user_question: User's question
        opensearch_kb: OpenSearch knowledge base client
        llm: LangChain LLM instance
        conversation_history: Current conversation history
    
    Returns:
        Tuple of (answer, used_knowledge_base)
    """
    used_kb = False
    context = ""
    
    if config.USE_KNOWLEDGE_BASE:
        # Search knowledge base
        print("🔍 Searching knowledge base...")
        documents = opensearch_kb.search_documents(
            query=user_question,
            top_k=config.TOP_K_DOCUMENTS,
            min_score=config.MIN_RELEVANCE_SCORE
        )
        
        if documents:
            used_kb = True
            print(f"✓ Found {len(documents)} relevant document(s)")
            context = opensearch_kb.format_context(documents)
        else:
            print("ℹ No relevant documents found, using general knowledge")
    
    # Create appropriate system prompt
    system_prompt = create_system_prompt(use_context=used_kb, context=context)
    
    # Build message history with the appropriate system prompt
    messages = [system_prompt] + conversation_history + [HumanMessage(content=user_question)]
    
    # Get LLM response
    response = llm.invoke(messages)
    cleaned_content = clean_response(response.content)
    
    return cleaned_content, used_kb


def main():
    """Main chatbot loop"""
    llm = create_llm()
    
    # Initialize OpenSearch knowledge base
    opensearch_kb = None
    if config.USE_KNOWLEDGE_BASE:
        try:
            print("Initializing OpenSearch knowledge base...")
            opensearch_kb = OpenSearchKnowledgeBase(
                host=config.OPENSEARCH_HOST,
                region=config.OPENSEARCH_REGION,
                index_name=config.OPENSEARCH_INDEX,
                port=config.OPENSEARCH_PORT,
                use_ssl=config.OPENSEARCH_USE_SSL,
                is_serverless=config.OPENSEARCH_IS_SERVERLESS
            )
            print("✓ OpenSearch connected successfully\n")
        except Exception as e:
            print(f"⚠ Warning: Could not connect to OpenSearch: {e}")
            print("Continuing with general knowledge only...\n")
            opensearch_kb = None
    
    # Conversation history (without system prompt - we'll add it per request)
    conversation_history = []
    
    print("=" * 60)
    print("RAG-Enhanced Chatbot Started")
    print("=" * 60)
    print("Your chatbot will search the knowledge base first,")
    print("then use general knowledge if needed.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break
        
        try:
            # Get answer with knowledge base search
            answer, used_kb = get_answer_with_knowledge_base(
                user_question=user_text,
                opensearch_kb=opensearch_kb if opensearch_kb else None,
                llm=llm,
                conversation_history=conversation_history
            )
            
            # Display answer with source indicator
            source_indicator = "📚 [From Knowledge Base]" if used_kb else "🤖 [General Knowledge]"
            print(f"\n{source_indicator}")
            print(f"Bot: {answer}\n")
            
            # Update conversation history (just the basic exchange)
            conversation_history.append(HumanMessage(content=user_text))
            conversation_history.append(AIMessage(content=answer))
            
            # Keep conversation history manageable (last 10 exchanges)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                
        except Exception as e:
            print(f"\n⚠ Error: {e}\n")


if __name__ == "__main__":
    main()
