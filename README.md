# MongoDB GraphRAG Architecture Design
## Migration from Neo4j to MongoDB with LangChain Integration

### Executive Summary

This document outlines a comprehensive architecture for migrating from Neo4j to MongoDB while implementing LangChain GraphRAG capabilities. The design focuses on a unified document structure that combines vector embeddings, text content, and graph relationships in a single MongoDB collection, optimized for multi-tenant environments.

---

## 1. Current Architecture Analysis

### 1.1 Existing Neo4j Implementation
Based on the codebase analysis, the current system:

- **Vector Storage**: Uses [`Neo4jVector`](app/services/vector_search_service.py:43) with tenant-specific indexes
- **Graph Storage**: Leverages [`LLMGraphTransformer`](app/tasks/ingestion.py:103) for entity/relationship extraction
- **Multi-tenancy**: Implements through `clientId`, `projectId`, `workspaceId` properties
- **Chunking**: Uses [`AgenticChunker`](app/tasks/agentic_chunker.py:11) for intelligent document segmentation
- **Search**: Separate [`graph_search_service`](app/services/graph_search_service.py:26) and [`vector_search_service`](app/services/vector_search_service.py:34)

### 1.2 Key Components to Migrate
- Document chunks with vector embeddings
- Knowledge graph entities and relationships
- Multi-tenant isolation
- Proposition extraction and agentic chunking
- Cypher-based graph queries

---

## 2. MongoDB Schema Design

### 2.1 Unified Collection Structure

#### Primary Collection: `rag_documents`

```javascript
{
  // Document Identification
  "_id": ObjectId("..."),
  "documentId": "uuid-v4-string",
  "chunkId": "uuid-v4-string", // From AgenticChunker
  "chunkIndex": 0,
  
  // Multi-tenant Organization
  "tenancy": {
    "clientId": "client-123",
    "projectId": "project-456", // Optional
    "workspaceId": "workspace-789" // Optional
  },
  
  // Content and Embeddings
  "content": {
    "text": "The actual chunk text content...",
    "propositions": [
      "Individual proposition 1",
      "Individual proposition 2"
    ],
    "summary": "AI-generated chunk summary",
    "title": "AI-generated chunk title"
  },
  
  // Vector Embeddings
  "embedding": [0.1, 0.2, 0.3, ...], // 1536-dimensional vector
  
  // Graph Data Embedded
  "graph": {
    "entities": [
      {
        "id": "entity-uuid",
        "label": "Person",
        "name": "John Doe",
        "type": "PERSON",
        "properties": {
          "age": 30,
          "occupation": "Engineer"
        },
        "mentions": [
          {
            "start": 45,
            "end": 53,
            "confidence": 0.95
          }
        ]
      }
    ],
    "relationships": [
      {
        "id": "rel-uuid",
        "type": "WORKS_FOR",
        "sourceEntityId": "entity-uuid-1",
        "targetEntityId": "entity-uuid-2",
        "properties": {
          "since": "2020-01-01",
          "role": "Senior Engineer"
        },
        "confidence": 0.88
      }
    ]
  },
  
  // Metadata
  "metadata": {
    "source": "document.pdf",
    "pageNumber": 1,
    "createdAt": ISODate("2024-01-01T00:00:00Z"),
    "updatedAt": ISODate("2024-01-01T00:00:00Z"),
    "version": 1
  },
  
  // Search Optimization
  "searchTerms": ["extracted", "keywords", "for", "text", "search"],
  "entityTypes": ["PERSON", "ORGANIZATION", "LOCATION"],
  "relationshipTypes": ["WORKS_FOR", "LOCATED_IN"]
}
```

### 2.2 Supporting Collections

#### Entity Index Collection: `entity_index`
```javascript
{
  "_id": ObjectId("..."),
  "entityId": "entity-uuid",
  "name": "John Doe",
  "type": "PERSON",
  "tenancy": {
    "clientId": "client-123",
    "projectId": "project-456",
    "workspaceId": "workspace-789"
  },
  "documentReferences": [
    {
      "documentId": "doc-uuid",
      "chunkId": "chunk-uuid",
      "mentions": 3,
      "confidence": 0.95
    }
  ],
  "properties": {
    "canonicalName": "John Doe",
    "aliases": ["J. Doe", "Johnny"],
    "firstSeen": ISODate("2024-01-01T00:00:00Z"),
    "lastSeen": ISODate("2024-01-01T00:00:00Z")
  }
}
```

#### Relationship Index Collection: `relationship_index`
```javascript
{
  "_id": ObjectId("..."),
  "relationshipId": "rel-uuid",
  "type": "WORKS_FOR",
  "sourceEntityId": "entity-uuid-1",
  "targetEntityId": "entity-uuid-2",
  "tenancy": {
    "clientId": "client-123",
    "projectId": "project-456",
    "workspaceId": "workspace-789"
  },
  "documentReferences": [
    {
      "documentId": "doc-uuid",
      "chunkId": "chunk-uuid",
      "confidence": 0.88
    }
  ],
  "properties": {
    "strength": 0.88,
    "frequency": 5,
    "firstSeen": ISODate("2024-01-01T00:00:00Z"),
    "lastSeen": ISODate("2024-01-01T00:00:00Z")
  }
}
```

---

## 3. Indexing Strategy

### 3.1 Vector Search Indexes (MongoDB Atlas Vector Search)

```javascript
// Vector Search Index for rag_documents collection
{
  "name": "vector_search_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1536,
        "similarity": "cosine"
      },
      {
        "type": "filter",
        "path": "tenancy.clientId"
      },
      {
        "type": "filter", 
        "path": "tenancy.projectId"
      },
      {
        "type": "filter",
        "path": "tenancy.workspaceId"
      },
      {
        "type": "filter",
        "path": "entityTypes"
      },
      {
        "type": "filter",
        "path": "relationshipTypes"
      }
    ]
  }
}
```

### 3.2 Graph Traversal Indexes

```javascript
// Compound indexes for efficient graph traversal
db.rag_documents.createIndex({
  "tenancy.clientId": 1,
  "graph.entities.id": 1,
  "graph.entities.type": 1
});

db.rag_documents.createIndex({
  "tenancy.clientId": 1,
  "graph.relationships.sourceEntityId": 1,
  "graph.relationships.type": 1
});

db.rag_documents.createIndex({
  "tenancy.clientId": 1,
  "graph.relationships.targetEntityId": 1,
  "graph.relationships.type": 1
});

// Entity index collection indexes
db.entity_index.createIndex({
  "tenancy.clientId": 1,
  "entityId": 1
});

db.entity_index.createIndex({
  "tenancy.clientId": 1,
  "type": 1,
  "name": 1
});

// Relationship index collection indexes
db.relationship_index.createIndex({
  "tenancy.clientId": 1,
  "sourceEntityId": 1,
  "type": 1
});

db.relationship_index.createIndex({
  "tenancy.clientId": 1,
  "targetEntityId": 1,
  "type": 1
});
```

### 3.3 Multi-tenancy and Search Optimization

```javascript
// Text search index
db.rag_documents.createIndex({
  "tenancy.clientId": 1,
  "content.text": "text",
  "searchTerms": "text"
});

// Metadata and temporal indexes
db.rag_documents.createIndex({
  "tenancy.clientId": 1,
  "metadata.createdAt": -1
});

db.rag_documents.createIndex({
  "tenancy.clientId": 1,
  "documentId": 1,
  "chunkIndex": 1
});
```

---

## 4. LangChain GraphRAG Integration

### 4.1 Custom MongoDB GraphRAG Retriever

```python
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.base import VectorStore
from typing import List, Dict, Any, Optional
import pymongo
from pymongo import MongoClient

class MongoDBGraphRAGRetriever(BaseRetriever):
    """
    Custom retriever that implements GraphRAG functionality with MongoDB.
    Combines vector similarity search with graph traversal.
    """
    
    def __init__(
        self,
        mongo_client: MongoClient,
        database_name: str,
        collection_name: str = "rag_documents",
        vector_index_name: str = "vector_search_index",
        tenancy_filter: Dict[str, str] = None,
        k: int = 5,
        graph_depth: int = 2
    ):
        self.mongo_client = mongo_client
        self.database = mongo_client[database_name]
        self.collection = self.database[collection_name]
        self.entity_index = self.database["entity_index"]
        self.relationship_index = self.database["relationship_index"]
        self.vector_index_name = vector_index_name
        self.tenancy_filter = tenancy_filter or {}
        self.k = k
        self.graph_depth = graph_depth
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid vector + graph approach.
        """
        # Step 1: Vector similarity search
        vector_results = self._vector_search(query)
        
        # Step 2: Extract entities from top vector results
        entities = self._extract_entities_from_results(vector_results)
        
        # Step 3: Graph traversal to find related entities
        expanded_entities = self._graph_traversal(entities)
        
        # Step 4: Retrieve documents containing expanded entities
        graph_results = self._get_documents_by_entities(expanded_entities)
        
        # Step 5: Combine and rank results
        combined_results = self._combine_and_rank_results(
            vector_results, graph_results, query
        )
        
        return combined_results[:self.k]
    
    def _vector_search(self, query: str) -> List[Dict]:
        """
        Perform vector similarity search using MongoDB Atlas Vector Search.
        """
        # This would use the embedding model to convert query to vector
        query_vector = self._embed_query(query)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": self.k * 3,
                    "limit": self.k,
                    "filter": self.tenancy_filter
                }
            },
            {
                "$addFields": {
                    "vectorScore": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        return list(self.collection.aggregate(pipeline))
    
    def _graph_traversal(self, seed_entities: List[str]) -> List[str]:
        """
        Perform graph traversal to find related entities.
        """
        visited_entities = set(seed_entities)
        current_entities = seed_entities.copy()
        
        for depth in range(self.graph_depth):
            next_entities = []
            
            # Find relationships from current entities
            pipeline = [
                {
                    "$match": {
                        **self.tenancy_filter,
                        "sourceEntityId": {"$in": current_entities}
                    }
                },
                {
                    "$group": {
                        "_id": "$targetEntityId",
                        "strength": {"$avg": "$properties.strength"},
                        "frequency": {"$sum": "$properties.frequency"}
                    }
                },
                {
                    "$sort": {"strength": -1, "frequency": -1}
                },
                {
                    "$limit": 10
                }
            ]
            
            relationships = list(self.relationship_index.aggregate(pipeline))
            
            for rel in relationships:
                target_entity = rel["_id"]
                if target_entity not in visited_entities:
                    next_entities.append(target_entity)
                    visited_entities.add(target_entity)
            
            current_entities = next_entities
            if not current_entities:
                break
        
        return list(visited_entities)
```

### 4.2 Integration with Existing AgenticChunker

```python
class MongoDBAgenticChunker(AgenticChunker):
    """
    Extended AgenticChunker that stores results directly in MongoDB
    with embedded graph data.
    """
    
    def __init__(self, mongo_client: MongoClient, database_name: str, 
                 tenancy_filter: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.mongo_client = mongo_client
        self.database = mongo_client[database_name]
        self.collection = self.database["rag_documents"]
        self.tenancy_filter = tenancy_filter
    
    def store_chunks_with_graph(self, document_id: str, 
                               graph_documents: List[Any]) -> List[str]:
        """
        Store chunks with embedded graph data in MongoDB.
        """
        stored_chunk_ids = []
        
        for chunk_id, chunk_data in self.chunks.items():
            # Extract entities and relationships for this chunk
            chunk_graph = self._extract_chunk_graph(
                chunk_data, graph_documents
            )
            
            # Generate embedding for chunk content
            chunk_text = " ".join(chunk_data['propositions'])
            embedding = self._generate_embedding(chunk_text)
            
            # Create MongoDB document
            mongo_doc = {
                "documentId": document_id,
                "chunkId": chunk_id,
                "chunkIndex": chunk_data['chunk_index'],
                "tenancy": self.tenancy_filter,
                "content": {
                    "text": chunk_text,
                    "propositions": chunk_data['propositions'],
                    "summary": chunk_data['summary'],
                    "title": chunk_data['title']
                },
                "embedding": embedding,
                "graph": chunk_graph,
                "metadata": {
                    "createdAt": datetime.utcnow(),
                    "version": 1
                },
                "searchTerms": self._extract_keywords(chunk_text),
                "entityTypes": list(set([e["type"] for e in chunk_graph["entities"]])),
                "relationshipTypes": list(set([r["type"] for r in chunk_graph["relationships"]]))
            }
            
            # Insert into MongoDB
            result = self.collection.insert_one(mongo_doc)
            stored_chunk_ids.append(
