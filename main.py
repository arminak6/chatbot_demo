from langchain_aws import ChatBedrock, AmazonKnowledgeBasesRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re

# Configuration
AWS_REGION = "eu-south-1" 
MODEL_ID = "openai.gpt-oss-20b-1:0"
KNOWLEDGE_BASE_ID = "79FFW5FF2S"  # Your Bedrock Knowledge Base ID

# System Prompt
SYSTEM_PROMPT_CONTENT = (
    "You are a professional AI assistant.\n\n"
    "Rules:\n"
    "- If 'Context from documents' is provided below, answer based PRIMARILY on that context.\n"
    "- If the context doesn't contain the answer, use your general knowledge.\n"
    "- Provide ONLY the final answer. Do NOT explain your reasoning.\n"
    "- Keep answers short, clear, and simple.\n"
    "- Never invent facts.\n"
    "- If you don't know the answer, say 'I don't know'."
)

def clean_response(text):
    """Remove reasoning tags and clean response"""
    cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def create_llm():
    """Create and configure the LLM"""
    return ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.1,  # Lower temperature for more factual answers
        },
    )

def create_retriever():
    """Create Bedrock Knowledge Base retriever"""
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        region_name=AWS_REGION,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 3  # Number of document chunks to retrieve
            }
        },
    )

def main():
    """Main chatbot loop"""
    print("Initializing chatbot with Bedrock Knowledge Base...")
    llm = create_llm()
    retriever = create_retriever()
    
    # Conversation history with system prompt
    history = [SystemMessage(content=SYSTEM_PROMPT_CONTENT)]
    
    print("=" * 60)
    print("RAG-Enhanced Chatbot Started")
    print("=" * 60)
    print(f"Using Knowledge Base: {KNOWLEDGE_BASE_ID}")
    print("Your chatbot will search documents first, then use general knowledge.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break
        
        try:
            # STEP 1: Retrieve from Knowledge Base
            print("🔍 Searching knowledge base...")
            relevant_docs = retriever.invoke(user_text)
            
            # Check if we got relevant documents
            if relevant_docs and len(relevant_docs) > 0:
                print(f"✓ Found {len(relevant_docs)} relevant document(s)")
                
                # Combine document contents
                context_text = "\n\n".join([
                    f"[Document {i+1}]\n{doc.page_content}" 
                    for i, doc in enumerate(relevant_docs)
                ])
                
                # STEP 2: Augment the prompt with context
                enriched_query = (
                    f"Context from documents:\n{context_text}\n\n"
                    f"User Question: {user_text}"
                )
                
                source_indicator = "📚 [From Knowledge Base]"
            else:
                print("ℹ No relevant documents found, using general knowledge")
                enriched_query = user_text
                source_indicator = "🤖 [General Knowledge]"
            
            # Add to history
            history.append(HumanMessage(content=enriched_query))
            
            # STEP 3: Generate response
            response = llm.invoke(history)
            cleaned_content = clean_response(response.content)
            
            # Display answer with source indicator
            print(f"\n{source_indicator}")
            print(f"Bot: {cleaned_content}\n")
            
            # Save AI response to history
            history.append(AIMessage(content=response.content))
            
            # Keep conversation history manageable (last 10 exchanges)
            if len(history) > 21:  # Keep system prompt + last 10 exchanges (20 messages)
                history = [history[0]] + history[-20:]
                
        except Exception as e:
            print(f"\n⚠ Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
