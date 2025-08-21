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

### Azure AI Search Index Schema
```json
{
  "name": "unified-rag-index",
  "fields": [
    // Common fields
    {"name": "id", "type": "Edm.String", "key": true},
    {"name": "content", "type": "Edm.String", "searchable": true},
    {"name": "content_vector", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": 3072},
    {"name": "document_type", "type": "Edm.String", "filterable": true}, // "chunk", "entity", "relationship"
    
    // Multi-tenancy fields
    {"name": "clientId", "type": "Edm.String", "filterable": true},
    {"name": "projectId", "type": "Edm.String", "filterable": true},
    {"name": "workspaceId", "type": "Edm.String", "filterable": true},
    
    // Document chunk fields
    {"name": "source", "type": "Edm.String", "filterable": true},
    {"name": "chunk_metadata", "type": "Edm.String"},
    
    // Graph entity fields
    {"name": "entity_id", "type": "Edm.String", "filterable": true},
    {"name": "entity_type", "type": "Edm.String", "filterable": true},
    {"name": "entity_description", "type": "Edm.String", "searchable": true},
    
    // Graph relationship fields
    {"name": "source_entity", "type": "Edm.String", "filterable": true},
    {"name": "target_entity", "type": "Edm.String", "filterable": true},
    {"name": "relationship_type", "type": "Edm.String", "filterable": true},
    {"name": "relationship_properties", "type": "Edm.String"}
  ]
}
```

### Document Types in Azure AI Search

#### 1. Document Chunks
```json
{
  "id": "chunk_client123_doc1_0",
  "content": "This is a text chunk from the document...",
  "content_vector": [0.1, 0.2, ...],
  "document_type": "chunk",
  "clientId": "client-123",
  "projectId": "project-456",
  "source": "document.pdf",
  "chunk_metadata": "{\"page\": 1, \"section\": \"introduction\"}"
}
```

#### 2. Graph Entities
```json
{
  "id": "entity_client123_person_john_doe",
  "content": "John Doe is a software engineer at TechCorp...",
  "content_vector": [0.3, 0.4, ...],
  "document_type": "entity",
  "clientId": "client-123",
  "entity_id": "john_doe",
  "entity_type": "Person",
  "entity_description": "Software engineer at TechCorp with 5 years experience"
}
```

#### 3. Graph Relationships
```json
{
  "id": "rel_client123_john_doe_works_at_techcorp",
  "content": "John Doe works at TechCorp as a software engineer",
  "content_vector": [0.5, 0.6, ...],
  "document_type": "relationship",
  "clientId": "client-123",
  "source_entity": "john_doe",
  "target_entity": "techcorp",
  "relationship_type": "WORKS_AT",
  "relationship_properties": "{\"role\": \"software_engineer\", \"since\": \"2019\"}"
}
```

## GraphRAG Implementation Strategy

### 1. Entity Extraction and Storage
```python
# During ingestion, extract entities and relationships
def extract_graph_from_document(document_content, user_data):
    # Use LLM to extract entities
    entities = llm_extract_entities(document_content)
    relationships = llm_extract_relationships(document_content, entities)
    
    # Store entities in Azure AI Search
    for entity in entities:
        entity_doc = {
            "id": f"entity_{user_data['client_id']}_{entity['id']}",
            "content": entity['description'],
            "document_type": "entity",
            "entity_id": entity['id'],
            "entity_type": entity['type'],
            "entity_description": entity['description'],
            **user_data  # clientId, projectId, workspaceId
        }
        azure_search.add_documents([entity_doc])
    
    # Store relationships in Azure AI Search
    for rel in relationships:
        rel_doc = {
            "id": f"rel_{user_data['client_id']}_{rel['source']}_{rel['type']}_{rel['target']}",
            "content": f"{rel['source']} {rel['type']} {rel['target']}",
            "document_type": "relationship",
            "source_entity": rel['source'],
            "target_entity": rel['target'],
            "relationship_type": rel['type'],
            **user_data
        }
        azure_search.add_documents([rel_doc])
```

### 2. GraphRAG Query Process
```python
# GraphRAG query implementation
def graphrag_query(query: str, user_data: Dict[str, Any]) -> str:
    # Step 1: Find relevant entities using vector search
    relevant_entities = azure_search.search(
        query=query,
        filter=f"document_type eq 'entity' and clientId eq '{user_data['client_id']}'",
        top=10
    )
    
    # Step 2: Find relationships for relevant entities
    entity_ids = [entity['entity_id'] for entity in relevant_entities]
    relationships = []
    for entity_id in entity_ids:
        rels = azure_search.search(
            filter=f"document_type eq 'relationship' and (source_entity eq '{entity_id}' or target_entity eq '{entity_id}') and clientId eq '{user_data['client_id']}'",
            top=20
        )
        relationships.extend(rels)
    
    # Step 3: Build context from entities and relationships
    context = build_graph_context(relevant_entities, relationships)
    
    # Step 4: Use LLM to answer query based on graph context
    answer = llm.invoke(f"""
    Based on the following graph information, answer the query: {query}
    
    Entities: {format_entities(relevant_entities)}
    Relationships: {format_relationships(relationships)}
    
    Answer:
    """)
    
    return answer
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

## Benefits of This Architecture

1. **Simplified Infrastructure**
   - Single Azure AI Search service instead of Neo4j
   - Reduced operational complexity
   - Better Azure ecosystem integration

2. **Enhanced Search Capabilities**
   - Unified vector and graph search
   - Better semantic search with Azure AI
   - Advanced filtering and faceting

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
