# RAG Chatbot with AWS Bedrock Knowledge Base

A simple chatbot that integrates AWS Bedrock with your Bedrock Knowledge Base to provide answers from your documents first, falling back to the model's general knowledge when needed.

## Features

- 🔍 **Semantic Search**: Uses AWS Bedrock Knowledge Bases vector search
- 🤖 **Fallback to LLM**: Uses general AI knowledge when documents don't have the answer
- 📊 **Source Indicators**: Shows whether answer came from knowledge base or general knowledge
- 💬 **Conversation History**: Maintains context across the conversation

## Quick Start

### 1. Configure

Edit [`main.py`](file:///c:/sinergia_ak/chatbot/SkillBuilder/chatbot_demo/main.py) and update these values:

```python
AWS_REGION = "eu-south-1"  # Your AWS region
MODEL_ID = "openai.gpt-oss-20b-1:0"  # Your Bedrock model
KNOWLEDGE_BASE_ID = "79FFW5FF2S"  # Your Knowledge Base ID
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

## How It Works

1. **User asks a question** → 
2. **Bedrock Knowledge Base Retriever** performs semantic vector search →
3. **If relevant documents found** → Adds them as context to the prompt →
4. **LLM generates answer** → Shows source indicator

## Example Usage

```
You: What is our return policy?
🔍 Searching knowledge base...
✓ Found 2 relevant document(s)

📚 [From Knowledge Base]
Bot: Our return policy allows 30 days for returns...

You: What's the weather?
🔍 Searching knowledge base...
ℹ No relevant documents found, using general knowledge

🤖 [General Knowledge]
Bot: I don't have real-time weather information.
```

## Configuration

### Number of Documents Retrieved

In `create_retriever()`:
```python
"vectorSearchConfiguration": {
    "numberOfResults": 3  # Adjust to retrieve more/fewer chunks
}
```

### Model Temperature

In `create_llm()`:
```python
"temperature": 0.1  # Lower = more factual, Higher = more creative
```

## AWS Setup Requirements

### 1. Bedrock Knowledge Base
- Create a Knowledge Base in AWS Bedrock
- Sync your documents
- Note the Knowledge Base ID

### 2. IAM Permissions

Your AWS credentials need:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:Retrieve"
      ],
      "Resource": "*"
    }
  ]
}
```

### 3. AWS Credentials

Ensure credentials are configured:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credentials file (`~/.aws/credentials`)
- IAM role (if on EC2/ECS)

## Troubleshooting

**Can't find Knowledge Base?**
- Verify `KNOWLEDGE_BASE_ID` is correct
- Check the Knowledge Base is in the same region as `AWS_REGION`

**No documents retrieved?**
- Ensure your Knowledge Base is synced
- Test the same question in AWS Bedrock playground
- Check documents are actually indexed

**Authentication errors?**
- Verify AWS credentials are configured
- Check IAM permissions include `bedrock:Retrieve`
