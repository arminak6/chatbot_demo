"""
OpenSearch client for retrieving knowledge base documents
"""
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from typing import List, Dict, Optional

class OpenSearchKnowledgeBase:
    def __init__(
        self,
        host: str,
        region: str = "eu-central-1",
        index_name: str = "knowledge_base",
        port: int = 443,
        use_ssl: bool = True,
        is_serverless: bool = False
    ):
        """
        Initialize OpenSearch client with AWS authentication
        
        Args:
            host: OpenSearch endpoint (without https://)
            region: AWS region
            index_name: Name of the index containing your documents
            port: OpenSearch port (default 443)
            use_ssl: Whether to use SSL (default True)
            is_serverless: True for OpenSearch Serverless, False for managed OpenSearch
        """
        self.index_name = index_name
        
        # Get AWS credentials
        credentials = boto3.Session().get_credentials()
        
        # Use 'aoss' for serverless, 'es' for managed OpenSearch
        service_name = 'aoss' if is_serverless else 'es'
        
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            service_name,
            session_token=credentials.token
        )
        
        # Create OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=awsauth,
            use_ssl=use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
    
    def search_documents(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.5
    ) -> List[Dict]:
        """
        Search for relevant documents in OpenSearch
        
        Args:
            query: User's question
            top_k: Number of top documents to retrieve
            min_score: Minimum relevance score (0-1)
        
        Returns:
            List of relevant documents with their content and scores
        """
        try:
            # Perform search query
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": top_k,
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content^2", "title", "metadata"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "min_score": min_score
                }
            )
            
            # Extract relevant documents
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'content': hit['_source'].get('content', ''),
                    'title': hit['_source'].get('title', ''),
                    'score': hit['_score'],
                    'metadata': hit['_source'].get('metadata', {})
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching OpenSearch: {e}")
            return []
    
    def format_context(self, documents: List[Dict]) -> str:
        """
        Format retrieved documents into context for the LLM
        
        Args:
            documents: List of documents from OpenSearch
        
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = ["Based on the following knowledge base documents:\n"]
        
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            score = doc.get('score', 0)
            
            context_parts.append(
                f"\n[Document {i}] {title} (Relevance: {score:.2f})\n{content}\n"
            )
        
        return "\n".join(context_parts)
