# Detailed Azure Migration Architecture

## Overview
This document outlines the detailed architecture for migrating the fast-rag-app from Neo4j to Azure AI Search using the GraphRAG approach. The solution uses Azure AI Search as the single storage backend for both vector search and graph relationships.

## Current vs. Target Architecture

### Current Architecture (Neo4j-based)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Vector Search │    │   Graph Search  │
│                 │    │                 │    │                 │
│ - Ingest API    │───▶│ Neo4jVector     │    │ GraphCypherQA   │
│ - Search API    │    │ - Embeddings    │    │ - Cypher Query  │
│ - Multi-tenant  │    │ - Chunks        │    │ - Relationships │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────────────────────────┐
                       │           Neo4j Database            │
                       │                                     │
                       │ - Vector Index (chunks)             │
                       │ - Graph Nodes (entities)            │
                       │ - Graph Relationships               │
                       │ - Multi-tenant isolation            │
                       └─────────────────────────────────────┘
```

### Target Architecture (Azure AI Search + GraphRAG)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Vector Search │    │   Graph Search  │
│                 │    │                 │    │                 │
│ - Ingest API    │───▶│ Azure AI Search │    │ GraphRAG        │
│ - Search API    │    │ - Embeddings    │    │ - Entity Search │
│ - Multi-tenant  │    │ - Chunks        │    │ - Relationship  │
└─────────────────┘    └─────────────────┘    │   Traversal     │
                                │              └─────────────────┘
                                ▼                        │
                       ┌─────────────────────────────────────┐
                       │        Azure AI Search Index       │
                       │                                     │
                       │ Document Types:                     │
                       │ 1. "chunk" - Text chunks            │
                       │ 2. "entity" - Graph entities        │
                       │ 3. "relationship" - Graph edges     │
                       │                                     │
                       │ Multi-tenant filtering via metadata │
                       └─────────────────────────────────────┘
```

## Data Model Design

### Azure AI Search Index Schema (Unified Document Chunks with Embedded Graph Information)
```json
{
  "name": "unified-rag-index",
  "fields": [
    // Core document fields
    {"name": "id", "type": "Edm.String", "key": true},
    {"name": "content", "type": "Edm.String", "searchable": true},
    {"name": "content_vector", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": 3072},
    
    // Multi-tenancy fields
    {"name": "clientId", "type": "Edm.String", "filterable": true},
    {"name": "projectId", "type": "Edm.String", "filterable": true},
    {"name": "workspaceId", "type": "Edm.String", "filterable": true},
    
    // Document metadata
    {"name": "source", "type": "Edm.String", "filterable": true},
    {"name": "metadata", "type": "Edm.String"},
    
    // Graph information embedded in chunks
    {"name": "entities", "type": "Edm.String", "searchable": true}, // JSON string of entities in this chunk
    {"name": "relationships", "type": "Edm.String", "searchable": true}, // JSON string of relationships in this chunk
    {"name": "entity_types", "type": "Edm.String", "filterable": true}, // Comma-separated entity types for filtering
    {"name": "relationship_types", "type": "Edm.String", "filterable": true} // Comma-separated relationship types for filtering
  ]
}
```

### Unified Document Chunks with Embedded Graph Information

#### Document Chunk with Embedded Graph Data
```json
{
  "id": "chunk_client123_doc1_0",
  "content": "John Doe is a software engineer at TechCorp. He has been working there since 2019 and specializes in backend development.",
  "content_vector": [0.1, 0.2, ...],
  "clientId": "client-123",
  "projectId": "project-456",
  "source": "document.pdf",
  "metadata": "{\"page\": 1, \"section\": \"team_overview\"}",
  "entities": "[{\"id\": \"john_doe\", \"type\": \"Person\", \"description\": \"Software engineer at TechCorp\"}, {\"id\": \"techcorp\", \"type\": \"Company\", \"description\": \"Technology company\"}]",
  "relationships": "[{\"source\": \"john_doe\", \"target\": \"techcorp\", \"type\": \"WORKS_AT\", \"properties\": {\"role\": \"software_engineer\", \"since\": \"2019\"}}]",
  "entity_types": "Person,Company",
  "relationship_types": "WORKS_AT"
}
```

## GraphRAG Implementation Strategy (Unified Approach)

### 1. Entity and Relationship Extraction with Document Chunks
```python
# During ingestion, extract entities and relationships and embed them with chunks
def extract_graph_from_document_chunks(chunks, user_data):
    """
    Extract entities and relationships from each chunk and store them together.
    This creates a unified storage where each chunk contains its graph information.
    """
    chunks_with_graph_info = []
    
    for chunk_text in chunks:
        # Use LLM to extract entities and relationships from this specific chunk
        entities = llm_extract_entities(chunk_text)
        relationships = llm_extract_relationships(chunk_text, entities)
        
        # Create graph information for this chunk
        graph_info = {
            "entities": entities,  # List of entities found in this chunk
            "relationships": relationships  # List of relationships found in this chunk
        }
        
        chunks_with_graph_info.append({
            "text": chunk_text,
            "graph_info": graph_info
        })
    
    return chunks_with_graph_info
```

### 2. GraphRAG Query Process (Chunk-Based Traversal)
```python
# GraphRAG query implementation using chunk traversal
def graphrag_query(query: str, user_data: Dict[str, Any]) -> str:
    # Step 1: Find relevant chunks using vector search
    relevant_chunks = azure_search.search_documents(
        query=query,
        user_data=user_data,
        top_k=10
    )
    
    # Step 2: Extract entities from relevant chunks
    all_entities = []
    for chunk in relevant_chunks:
        entities = json.loads(chunk.get("entities", "[]"))
        all_entities.extend(entities)
    
    # Step 3: Find additional chunks that contain related entities
    entity_ids = [entity["id"] for entity in all_entities]
    related_chunks = azure_search.get_related_chunks(
        entity_ids=entity_ids,
        user_data=user_data,
        top_k=20
    )
    
    # Step 4: Combine all chunks and build comprehensive context
    all_chunks = relevant_chunks + related_chunks
    
    # Step 5: Build graph context from all chunks
    context = build_graph_context_from_chunks(all_chunks)
    
    # Step 6: Use LLM to answer query based on graph context
    answer = llm.invoke(f"""
    Based on the following information from document chunks and their embedded graph data, answer the query: {query}
    
    Context: {context}
    
    Answer:
    """)
    
    return answer

def build_graph_context_from_chunks(chunks):
    """
    Build a comprehensive graph context from chunks with embedded graph information.
    """
    all_entities = []
    all_relationships = []
    chunk_contents = []
    
    for chunk in chunks:
        chunk_contents.append(chunk["content"])
        
        # Parse entities and relationships from this chunk
        entities = json.loads(chunk.get("entities", "[]"))
        relationships = json.loads(chunk.get("relationships", "[]"))
        
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    
    # Deduplicate entities and relationships
    unique_entities = {entity["id"]: entity for entity in all_entities}.values()
    unique_relationships = list({f"{rel['source']}_{rel['type']}_{rel['target']}": rel for rel in all_relationships}.values())
    
    return {
        "chunks": chunk_contents,
        "entities": list(unique_entities),
        "relationships": unique_relationships
    }
```

## Implementation Plan

### Phase 1: Azure AI Search Integration
1. **Create Azure AI Search Client** (`app/db/azure_search_client.py`)
   - Unified client for chunks, entities, and relationships
   - Multi-tenant index management
   - Vector search capabilities

2. **Update Vector Search Service** (`app/services/vector_search_service.py`)
   - Replace Neo4jVector with Azure AI Search
   - Maintain existing API interface
   - Preserve tenant filtering

3. **Update Configuration** (`app/core/config.py`)
   - Add Azure AI Search settings
   - Remove Neo4j settings
   - Add migration flags

4. **Update Dependencies** (`requirements.txt`)
   - Add Azure AI Search packages
   - Remove Neo4j packages

### Phase 2: GraphRAG Implementation
1. **Create GraphRAG Service** (`app/services/graphrag_service.py`)
   - Entity extraction and storage
   - Relationship extraction and storage
   - Graph traversal using vector search

2. **Update Graph Search Service** (`app/services/graph_search_service.py`)
   - Replace Cypher queries with GraphRAG
   - Maintain existing API interface
   - Implement multi-hop relationship traversal

3. **Update Ingestion Process** (`app/tasks/ingestion.py`)
   - Extract and store entities alongside chunks
   - Extract and store relationships
   - Maintain proposition-based chunking

### Phase 3: Configuration and Deployment
1. **Environment Configuration**
   - Azure service credentials
   - Index configuration
   - Migration switches

2. **Deployment Updates**
   - Docker configuration
   - Environment variables
   - Health checks

## Migration Strategy

### Gradual Migration Approach
```python
# Configuration flag to enable gradual migration
class Settings(BaseSettings):
    # Migration flags
    USE_AZURE_SEARCH: bool = False  # Start with False
    USE_GRAPHRAG: bool = False      # Start with False
    
    # Fallback to Neo4j when Azure is disabled
    ENABLE_NEO4J_FALLBACK: bool = True
```

### Migration Steps
1. **Deploy Azure AI Search alongside Neo4j**
   - Set `USE_AZURE_SEARCH = False`
   - Test Azure AI Search in parallel

2. **Enable Vector Search Migration**
   - Set `USE_AZURE_SEARCH = True`
   - Keep `USE_GRAPHRAG = False`
   - Vector queries go to Azure, graph queries to Neo4j

3. **Enable Full GraphRAG**
   - Set `USE_GRAPHRAG = True`
   - All queries go to Azure AI Search
   - Disable Neo4j fallback

## Benefits of This Unified Architecture

1. **Simplified Infrastructure**
   - Single Azure AI Search service for both vector and graph operations
   - No separate graph database required
   - Reduced operational complexity
   - Better Azure ecosystem integration

2. **Unified Data Model**
   - Document chunks contain both content and graph information
   - Consistent multi-tenancy across vector and graph search
   - Simplified data ingestion and management
   - Single source of truth for all data

3. **Enhanced Search Capabilities**
   - Seamless transition between vector and graph search
   - Graph traversal through chunk relationships
   - Better semantic search with Azure AI
   - Advanced filtering and faceting on both content and graph data

3. **Improved Scalability**
   - Azure AI Search auto-scaling
   - Better performance for large datasets
   - Reduced memory requirements

4. **Cost Optimization**
   - Pay-per-use Azure AI Search
   - No dedicated graph database costs
   - Better resource utilization

5. **GraphRAG Advantages**
   - More flexible than rigid Cypher queries
   - Better handling of complex relationships
   - LLM-powered graph reasoning
   - Natural language graph traversal

## API Compatibility

The migration maintains full backward compatibility:

- **Vector Search Endpoint**: `/vector-search` - Same request/response format
- **Graph Search Endpoint**: `/graph-search` - Same request/response format  
- **Ingestion Endpoint**: `/ingest` - Same request/response format
- **Multi-tenancy**: Same tenant isolation mechanism
- **Authentication**: No changes to JWT authentication

This ensures zero downtime migration and no client-side changes required.
