"""
Configuration file for chatbot and OpenSearch settings
"""

# AWS Settings
AWS_REGION = "eu-central-1"  # Region for Bedrock LLM

# LLM Settings
MODEL_ID = "openai.gpt-oss-20b-1:0"
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# OpenSearch Serverless Settings
OPENSEARCH_HOST = "jy1oprnu7bfueumws6zj.eu-south-1.aoss.amazonaws.com"  # Without https://
OPENSEARCH_REGION = "eu-south-1"  # Region for OpenSearch Serverless
OPENSEARCH_INDEX = "bedrock-knowledge-base-default-index"  # AWS Bedrock Knowledge Base index
OPENSEARCH_PORT = 443
OPENSEARCH_USE_SSL = True
OPENSEARCH_IS_SERVERLESS = True  # Set to True for serverless, False for managed

# RAG Settings
TOP_K_DOCUMENTS = 3  # Number of documents to retrieve
MIN_RELEVANCE_SCORE = 0.5  # Minimum score for considering a document relevant (0-1)
USE_KNOWLEDGE_BASE = True  # Set to False to disable knowledge base search
