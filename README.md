# RAG-Enhanced Chatbot with AWS OpenSearch

A chatbot that integrates AWS Bedrock with OpenSearch to provide answers from your knowledge base first, falling back to the model's general knowledge when needed.

## Features

- 🔍 **Knowledge Base Search**: Automatically searches your OpenSearch documents
- 🤖 **Fallback to LLM**: Uses general AI knowledge when documents don't have the answer
- 📊 **Source Indicators**: Shows whether answer came from knowledge base or general knowledge
- 💬 **Conversation History**: Maintains context across the conversation

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenSearch

Edit `config.py` and update the following settings:

```python
# Your OpenSearch domain endpoint (without https://)
OPENSEARCH_HOST = "your-domain.eu-central-1.es.amazonaws.com"

# Your index name
OPENSEARCH_INDEX = "your_index_name"

# Optional: Adjust these parameters
TOP_K_DOCUMENTS = 3  # Number of documents to retrieve
MIN_RELEVANCE_SCORE = 0.5  # Minimum relevance score (0-1)
```

### 3. AWS Credentials

Ensure your AWS credentials are configured:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credentials file (`~/.aws/credentials`)
- IAM role (if running on EC2/ECS)

### 4. Required Permissions

Your AWS user/role needs:
- **Bedrock**: Access to invoke the model
- **OpenSearch**: Read permissions on your domain and index

Example IAM policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost"
      ],
      "Resource": "arn:aws:es:eu-central-1:YOUR_ACCOUNT:domain/YOUR_DOMAIN/*"
    }
  ]
}
```

## Usage

Run the chatbot:

```bash
python main.py
```

The chatbot will:
1. Search your OpenSearch knowledge base for relevant documents
2. If relevant documents are found (score ≥ 0.5), answer based on those documents
3. If no relevant documents, use the LLM's general knowledge
4. Clearly indicate the source of each answer

Example interaction:
```
You: What is our refund policy?
🔍 Searching knowledge base...
✓ Found 2 relevant document(s)

📚 [From Knowledge Base]
Bot: We offer a 30-day money-back guarantee on all products...

You: What's the capital of France?
🔍 Searching knowledge base...
ℹ No relevant documents found, using general knowledge

🤖 [General Knowledge]
Bot: Paris is the capital of France.
```

## OpenSearch Index Structure

Your OpenSearch documents should have this structure:

```json
{
  "title": "Document Title",
  "content": "The main content of your document...",
  "metadata": {
    "category": "optional",
    "date": "optional"
  }
}
```

The search will prioritize the `content` field, but also search `title` and `metadata`.

## Configuration Options

In `config.py`:

- `USE_KNOWLEDGE_BASE`: Set to `False` to disable knowledge base search
- `TOP_K_DOCUMENTS`: Number of documents to retrieve (default: 3)
- `MIN_RELEVANCE_SCORE`: Minimum score to consider a document relevant (0-1, default: 0.5)
- `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`: LLM generation parameters

## Customization

### Adjusting Search Strategy

Edit `opensearch_client.py` to customize the search query. Current implementation uses `multi_match` with fuzzy matching. You can modify:

- Fields to search
- Field boosting (e.g., `content^2` gives content 2x weight)
- Fuzzy matching settings
- Query type (best_fields, most_fields, cross_fields, etc.)

### Custom System Prompts

Edit the `create_system_prompt()` function in `main.py` to customize how the bot uses context from documents.

## Troubleshooting

**Can't connect to OpenSearch?**
- Verify your `OPENSEARCH_HOST` is correct (without `https://`)
- Check AWS credentials have OpenSearch permissions
- Ensure your OpenSearch domain's access policy allows your AWS user/role

**No documents found?**
- Check your `OPENSEARCH_INDEX` name is correct
- Try lowering `MIN_RELEVANCE_SCORE`
- Verify documents exist in your index: `GET /your_index/_count`

**LLM not responding?**
- Verify Bedrock permissions
- Check `MODEL_ID` is available in your region
- Ensure model access is enabled in AWS Bedrock console
