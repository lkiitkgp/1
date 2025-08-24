# FastAPI RAG Application - Comprehensive Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Component Architecture](#component-architecture)
4. [Database Schema Design](#database-schema-design)
5. [API Flow Diagrams](#api-flow-diagrams)
6. [Data Flow Diagrams](#data-flow-diagrams)
7. [Sequence Diagrams](#sequence-diagrams)
8. [Multi-Tenant Architecture](#multi-tenant-architecture)
9. [Technology Stack](#technology-stack)
10. [Deployment Architecture](#deployment-architecture)
11. [Security Architecture](#security-architecture)
12. [Performance Considerations](#performance-considerations)

---

## Executive Summary

The FastAPI RAG (Retrieval-Augmented Generation) application is an enterprise-level document ingestion and retrieval system that combines vector similarity search with graph-based knowledge retrieval. The system features a sophisticated multi-tenant architecture, JWT-based authentication, and intelligent document processing with agentic chunking.

### Key Features
- **Dual Search Capabilities**: Vector similarity search using pgvector and graph-based search using Apache AGE
- **Multi-Tenant Architecture**: Hierarchical tenant isolation with client/project/workspace scoping
- **Intelligent Document Processing**: Proposition extraction and agentic chunking for optimal retrieval
- **Background Processing**: Asynchronous document ingestion with status tracking
- **Enterprise Security**: JWT authentication with RS256 algorithm and tenant validation

---

## System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Applications]
        API_CLIENT[API Clients]
        MOBILE[Mobile Apps]
    end
    
    subgraph "API Gateway Layer"
        LB[Load Balancer]
        CORS[CORS Middleware]
        AUTH[Authentication Middleware]
        TENANT[Tenant Context Middleware]
    end
    
    subgraph "FastAPI Application"
        MAIN[Main Application]
        ROUTER[API Router v1]
        
        subgraph "API Endpoints"
            INGEST_EP[Ingest Endpoints]
            SEARCH_EP[Search Endpoints]
            TENANT_EP[Tenant Endpoints]
            HEALTH_EP[Health Endpoints]
        end
        
        subgraph "Service Layer"
            VECTOR_SVC[Vector Search Service]
            GRAPH_SVC[Graph Search Service]
            TENANT_SVC[Tenant Service]
        end
        
        subgraph "Background Tasks"
            INGESTION[Document Ingestion]
            CHUNKER[Agentic Chunker]
            GRAPH_EXTRACT[Graph Extraction]
        end
    end
    
    subgraph "Database Layer"
        subgraph "PostgreSQL Database"
            PGVECTOR[(pgvector Extension)]
            AGE[(Apache AGE Extension)]
            METADATA[(Metadata Tables)]
        end
    end
    
    subgraph "External Services"
        OPENAI[OpenAI Compatible API]
        EMBEDDING[Embedding Service]
        LLM[LLM Service]
    end
    
    WEB --> LB
    API_CLIENT --> LB
    MOBILE --> LB
    
    LB --> CORS
    CORS --> AUTH
    AUTH --> TENANT
    TENANT --> MAIN
    
    MAIN --> ROUTER
    ROUTER --> INGEST_EP
    ROUTER --> SEARCH_EP
    ROUTER --> TENANT_EP
    ROUTER --> HEALTH_EP
    
    INGEST_EP --> VECTOR_SVC
    INGEST_EP --> INGESTION
    SEARCH_EP --> VECTOR_SVC
    SEARCH_EP --> GRAPH_SVC
    
    VECTOR_SVC --> TENANT_SVC
    GRAPH_SVC --> TENANT_SVC
    
    INGESTION --> CHUNKER
    INGESTION --> GRAPH_EXTRACT
    
    VECTOR_SVC --> PGVECTOR
    GRAPH_SVC --> AGE
    TENANT_SVC --> METADATA
    
    CHUNKER --> OPENAI
    GRAPH_EXTRACT --> LLM
    VECTOR_SVC --> EMBEDDING
```

### Architecture Principles
- **Layered Architecture**: Clear separation between API, service, and data layers
- **Microservice-Ready**: Modular design that can be easily decomposed
- **Tenant Isolation**: Comprehensive multi-tenancy with data segregation
- **Async Processing**: Background tasks for resource-intensive operations
- **Extensibility**: Plugin architecture for different LLM providers

---

## Component Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        FASTAPI[FastAPI Application]
        MIDDLEWARE[Middleware Stack]
        ENDPOINTS[API Endpoints]
    end
    
    subgraph "Business Logic Layer"
        subgraph "Core Services"
            VS[Vector Search Service]
            GS[Graph Search Service]
            TS[Tenant Service]
        end
        
        subgraph "Processing Services"
            DOC_PROC[Document Processor]
            CHUNK_SVC[Chunking Service]
            GRAPH_PROC[Graph Processor]
        end
        
        subgraph "Background Tasks"
            INGEST_TASK[Ingestion Task]
            STATUS_TRACK[Status Tracking]
        end
    end
    
    subgraph "Data Access Layer"
        DB_CLIENT[Database Client]
        VECTOR_CLIENT[Vector Client]
        GRAPH_CLIENT[Graph Client]
    end
    
    subgraph "Infrastructure Layer"
        CONFIG[Configuration Management]
        SECURITY[Security Module]
        LOGGING[Logging System]
    end
    
    subgraph "External Integration Layer"
        LLM_CLIENT[LLM Client]
        EMBED_CLIENT[Embedding Client]
        FILE_PROC[File Processor]
    end
    
    FASTAPI --> MIDDLEWARE
    MIDDLEWARE --> ENDPOINTS
    ENDPOINTS --> VS
    ENDPOINTS --> GS
    ENDPOINTS --> TS
    
    VS --> DOC_PROC
    GS --> GRAPH_PROC
    DOC_PROC --> CHUNK_SVC
    
    ENDPOINTS --> INGEST_TASK
    INGEST_TASK --> STATUS_TRACK
    
    VS --> DB_CLIENT
    GS --> DB_CLIENT
    DB_CLIENT --> VECTOR_CLIENT
    DB_CLIENT --> GRAPH_CLIENT
    
    CHUNK_SVC --> LLM_CLIENT
    GRAPH_PROC --> LLM_CLIENT
    VS --> EMBED_CLIENT
    INGEST_TASK --> FILE_PROC
    
    FASTAPI --> CONFIG
    FASTAPI --> SECURITY
    FASTAPI --> LOGGING
```

### Component Responsibilities

#### Presentation Layer
- **FastAPI Application**: Main application entry point and routing
- **Middleware Stack**: CORS, authentication, tenant context, and logging
- **API Endpoints**: RESTful endpoints for ingestion, search, and management

#### Business Logic Layer
- **Vector Search Service**: Semantic similarity search using embeddings
- **Graph Search Service**: Knowledge graph querying with Cypher
- **Tenant Service**: Multi-tenant context management and validation
- **Document Processor**: File parsing and content extraction
- **Chunking Service**: Intelligent document segmentation
- **Graph Processor**: Knowledge graph extraction from documents

#### Data Access Layer
- **Database Client**: Unified PostgreSQL connection management
- **Vector Client**: pgvector operations and collection management
- **Graph Client**: Apache AGE graph operations and query execution

---

## Database Schema Design

### Vector Database Schema (pgvector)

```mermaid
erDiagram
    langchain_pg_collection {
        uuid uuid PK
        string name UK
        jsonb cmetadata
        timestamp created_at
        timestamp updated_at
    }
    
    langchain_pg_embedding {
        uuid uuid PK
        uuid collection_id FK
        text document
        jsonb cmetadata
        vector embedding
        timestamp created_at
    }
    
    documents {
        uuid id PK
        uuid client_id
        uuid project_id
        uuid workspace_id
        text content
        vector embedding
        jsonb metadata
        string source_file
        integer chunk_index
        timestamp created_at
        timestamp updated_at
    }
    
    langchain_pg_collection ||--o{ langchain_pg_embedding : contains
    documents }o--|| langchain_pg_collection : stored_in
```

### Graph Database Schema (Apache AGE)

```mermaid
erDiagram
    ag_graph {
        oid oid PK
        string name UK
        string namespace
    }
    
    ag_label {
        oid oid PK
        string name
        oid graph FK
        integer id
        string kind
    }
    
    nodes {
        string id PK
        string node_type
        jsonb properties
        uuid client_id
        uuid project_id
        uuid workspace_id
        timestamp created_at
    }
    
    relationships {
        uuid id PK
        string source_node_id FK
        string target_node_id FK
        string relationship_type
        jsonb properties
        uuid client_id
        uuid project_id
        uuid workspace_id
        timestamp created_at
    }
    
    ag_graph ||--o{ ag_label : contains
    nodes }o--|| ag_label : labeled_by
    relationships }o--|| nodes : source
    relationships }o--|| nodes : target
```

### Multi-Tenant Data Model

```mermaid
erDiagram
    tenant_context {
        uuid client_id PK
        uuid project_id
        uuid workspace_id
        string collection_name
        string graph_name
        jsonb filter_dict
    }
    
    vector_collections {
        string name PK
        uuid client_id
        uuid project_id
        uuid workspace_id
        integer document_count
        timestamp created_at
    }
    
    graph_instances {
        string name PK
        uuid client_id
        uuid project_id
        uuid workspace_id
        integer node_count
        integer relationship_count
        timestamp created_at
    }
    
    tenant_context ||--o{ vector_collections : owns
    tenant_context ||--o{ graph_instances : owns
```

---

## API Flow Diagrams

### Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant Middleware
    participant Security
    participant TenantService
    participant Database
    
    Client->>Middleware: Request with JWT + Headers
    Middleware->>Security: Extract JWT Token
    Security->>Security: Decode & Validate JWT
    Security->>TenantService: Create Tenant Context
    TenantService->>TenantService: Validate UUIDs
    TenantService->>Security: Return Tenant Context
    Security->>Middleware: Return User Data
    Middleware->>Client: Proceed to Endpoint
```

### Ingestion API Flow

```mermaid
sequenceDiagram
    participant Client
    participant IngestEndpoint
    participant BackgroundTask
    participant DocumentProcessor
    participant VectorService
    participant GraphService
    participant Database
    
    Client->>IngestEndpoint: POST /ingest/ (file + metadata)
    IngestEndpoint->>DocumentProcessor: Parse uploaded file
    DocumentProcessor->>IngestEndpoint: Return document content
    IngestEndpoint->>BackgroundTask: Queue ingestion task
    IngestEndpoint->>Client: Return task_id (202)
    
    BackgroundTask->>DocumentProcessor: Extract propositions
    BackgroundTask->>VectorService: Create embeddings & store
    VectorService->>Database: Store in pgvector
    BackgroundTask->>GraphService: Extract knowledge graph
    GraphService->>Database: Store in Apache AGE
    BackgroundTask->>BackgroundTask: Update task status
```

### Search API Flow

```mermaid
sequenceDiagram
    participant Client
    participant SearchEndpoint
    participant VectorService
    participant GraphService
    participant Database
    participant LLM
    
    alt Vector Search
        Client->>SearchEndpoint: POST /search/vector-search
        SearchEndpoint->>VectorService: Query with tenant context
        VectorService->>Database: Similarity search in pgvector
        Database->>VectorService: Return similar documents
        VectorService->>LLM: Generate answer
        LLM->>VectorService: Return generated response
        VectorService->>SearchEndpoint: Return search result
        SearchEndpoint->>Client: Return response
    else Graph Search
        Client->>SearchEndpoint: POST /search/graph-search
        SearchEndpoint->>GraphService: Query with tenant context
        GraphService->>LLM: Generate Cypher query
        LLM->>GraphService: Return Cypher query
        GraphService->>Database: Execute Cypher in Apache AGE
        Database->>GraphService: Return graph results
        GraphService->>LLM: Generate natural language answer
        LLM->>GraphService: Return final answer
        GraphService->>SearchEndpoint: Return search result
        SearchEndpoint->>Client: Return response
    end
```

---

## Data Flow Diagrams

### Document Ingestion Pipeline

```mermaid
flowchart TD
    START([Document Upload]) --> PARSE[Parse Document]
    PARSE --> EXTRACT[Extract Text Content]
    EXTRACT --> PROPS[Generate Propositions]
    PROPS --> CHUNK[Agentic Chunking]
    
    CHUNK --> EMBED[Generate Embeddings]
    EMBED --> VECTOR_STORE[(Store in pgvector)]
    
    CHUNK --> GRAPH_EXTRACT[Extract Knowledge Graph]
    GRAPH_EXTRACT --> SANITIZE[Sanitize Labels]
    SANITIZE --> GRAPH_STORE[(Store in Apache AGE)]
    
    VECTOR_STORE --> COMPLETE[Mark Complete]
    GRAPH_STORE --> COMPLETE
    COMPLETE --> END([Ingestion Complete])
    
    subgraph "Tenant Context"
        TENANT[Client/Project/Workspace IDs]
    end
    
    TENANT -.-> VECTOR_STORE
    TENANT -.-> GRAPH_STORE
```

### Vector Search Workflow

```mermaid
flowchart TD
    QUERY([User Query]) --> VALIDATE[Validate Tenant Access]
    VALIDATE --> EMBED_QUERY[Generate Query Embedding]
    EMBED_QUERY --> FILTER[Apply Tenant Filters]
    FILTER --> SIMILARITY[Similarity Search]
    SIMILARITY --> RETRIEVE[Retrieve Documents]
    RETRIEVE --> CONTEXT[Build Context]
    CONTEXT --> LLM_GENERATE[Generate Answer]
    LLM_GENERATE --> RESPONSE([Return Response])
    
    subgraph "pgvector Database"
        COLLECTIONS[(Vector Collections)]
        EMBEDDINGS[(Document Embeddings)]
    end
    
    SIMILARITY --> COLLECTIONS
    COLLECTIONS --> EMBEDDINGS
    EMBEDDINGS --> RETRIEVE
```

### Graph Search Workflow

```mermaid
flowchart TD
    QUERY([User Query]) --> VALIDATE[Validate Tenant Access]
    VALIDATE --> SCHEMA[Get Graph Schema]
    SCHEMA --> CYPHER_GEN[Generate Cypher Query]
    CYPHER_GEN --> TENANT_FILTER[Add Tenant Filters]
    TENANT_FILTER --> EXECUTE[Execute Cypher]
    EXECUTE --> RESULTS[Process Results]
    RESULTS --> LLM_ANSWER[Generate Natural Language Answer]
    LLM_ANSWER --> RESPONSE([Return Response])
    
    subgraph "Apache AGE Database"
        GRAPHS[(Graph Instances)]
        NODES[(Nodes)]
        RELATIONSHIPS[(Relationships)]
    end
    
    EXECUTE --> GRAPHS
    GRAPHS --> NODES
    GRAPHS --> RELATIONSHIPS
    NODES --> RESULTS
    RELATIONSHIPS --> RESULTS
```

---

## Sequence Diagrams

### Document Ingestion Process

```mermaid
sequenceDiagram
    participant User
    participant API
    participant BackgroundTask
    participant AgenticChunker
    participant VectorDB
    participant GraphExtractor
    participant GraphDB
    participant LLM
    
    User->>API: Upload Document
    API->>BackgroundTask: Queue Ingestion Task
    API->>User: Return Task ID
    
    BackgroundTask->>LLM: Extract
Propositions
    LLM->>BackgroundTask: Return Propositions
    BackgroundTask->>AgenticChunker: Process Propositions
    AgenticChunker->>LLM: Generate Chunk Summaries
    LLM->>AgenticChunker: Return Summaries
    AgenticChunker->>BackgroundTask: Return Chunks
    
    BackgroundTask->>VectorDB: Store Embeddings
    VectorDB->>BackgroundTask: Confirm Storage
    
    BackgroundTask->>GraphExtractor: Extract Knowledge Graph
    GraphExtractor->>LLM: Process Document
    LLM->>GraphExtractor: Return Graph Elements
    GraphExtractor->>GraphDB: Store Nodes & Relationships
    GraphDB->>BackgroundTask: Confirm Storage
    
    BackgroundTask->>API: Update Task Status
```

### Vector Search Operation

```mermaid
sequenceDiagram
    participant User
    participant API
    participant VectorService
    participant TenantService
    participant VectorDB
    participant LLM
    
    User->>API: Search Query
    API->>TenantService: Validate Tenant Context
    TenantService->>API: Return Tenant Context
    API->>VectorService: Execute Search
    
    VectorService->>VectorDB: Check Collection Exists
    VectorDB->>VectorService: Confirm Collection
    VectorService->>VectorDB: Similarity Search with Filters
    VectorDB->>VectorService: Return Similar Documents
    
    VectorService->>LLM: Generate Answer from Context
    LLM->>VectorService: Return Generated Answer
    VectorService->>API: Return Search Results
    API->>User: Return Response
```

### Graph Search Operation

```mermaid
sequenceDiagram
    participant User
    participant API
    participant GraphService
    participant TenantService
    participant GraphDB
    participant LLM
    
    User->>API: Search Query
    API->>TenantService: Validate Tenant Context
    TenantService->>API: Return Tenant Context
    API->>GraphService: Execute Graph Search
    
    GraphService->>GraphDB: Refresh Graph Schema
    GraphDB->>GraphService: Return Schema
    GraphService->>LLM: Generate Cypher Query
    LLM->>GraphService: Return Cypher Query
    
    GraphService->>GraphDB: Execute Cypher with Tenant Filters
    GraphDB->>GraphService: Return Graph Results
    GraphService->>LLM: Generate Natural Language Answer
    LLM->>GraphService: Return Final Answer
    
    GraphService->>API: Return Search Results
    API->>User: Return Response
```

### Tenant Validation Process

```mermaid
sequenceDiagram
    participant Request
    participant Middleware
    participant Security
    participant TenantService
    participant JWT
    
    Request->>Middleware: HTTP Request with Headers
    Middleware->>Security: Extract JWT & Headers
    Security->>JWT: Decode JWT Token
    JWT->>Security: Return Payload
    
    Security->>TenantService: Validate Client ID
    TenantService->>TenantService: Check UUID Format
    TenantService->>Security: Return Validation Result
    
    Security->>TenantService: Create Tenant Context
    TenantService->>TenantService: Build Context Object
    TenantService->>Security: Return Tenant Context
    
    Security->>Middleware: Return User Data
    Middleware->>Request: Proceed with Validated Context
```

---

## Multi-Tenant Architecture

### Tenant Isolation Model

```mermaid
graph TB
    subgraph "Tenant Hierarchy"
        CLIENT[Client Level]
        PROJECT[Project Level]
        WORKSPACE[Workspace Level]
        
        CLIENT --> PROJECT
        PROJECT --> WORKSPACE
    end
    
    subgraph "Data Isolation"
        subgraph "Vector Collections"
            VC1[vector_index_client1]
            VC2[vector_index_client1_proj_project1]
            VC3[vector_index_client1_proj_project1_work_workspace1]
        end
        
        subgraph "Graph Instances"
            GI1[graph_client1]
            GI2[graph_client1_proj_project1]
            GI3[graph_client1_proj_project1_work_workspace1]
        end
        
        subgraph "Metadata Filtering"
            FILTER1[client_id filter]
            FILTER2[client_id + project_id filter]
            FILTER3[client_id + project_id + workspace_id filter]
        end
    end
    
    CLIENT -.-> VC1
    CLIENT -.-> GI1
    CLIENT -.-> FILTER1
    
    PROJECT -.-> VC2
    PROJECT -.-> GI2
    PROJECT -.-> FILTER2
    
    WORKSPACE -.-> VC3
    WORKSPACE -.-> GI3
    WORKSPACE -.-> FILTER3
```

### Tenant Context Flow

```mermaid
flowchart TD
    JWT[JWT Token] --> EXTRACT[Extract Client ID]
    HEADERS[HTTP Headers] --> EXTRACT_H[Extract Project/Workspace IDs]
    
    EXTRACT --> VALIDATE[Validate UUID Format]
    EXTRACT_H --> VALIDATE_H[Validate UUID Formats]
    
    VALIDATE --> CREATE[Create Tenant Context]
    VALIDATE_H --> CREATE
    
    CREATE --> COLLECTION[Generate Collection Name]
    CREATE --> GRAPH[Generate Graph Name]
    CREATE --> FILTER[Generate Filter Dict]
    
    COLLECTION --> VECTOR_OPS[Vector Operations]
    GRAPH --> GRAPH_OPS[Graph Operations]
    FILTER --> METADATA_FILTER[Metadata Filtering]
    
    subgraph "Naming Convention"
        CNAME[vector_index_clientid_proj_projectid_work_workspaceid]
        GNAME[graph_clientid_proj_projectid_work_workspaceid]
    end
    
    COLLECTION -.-> CNAME
    GRAPH -.-> GNAME
```

### Multi-Tenant Security Model

```mermaid
graph TB
    subgraph "Authentication Layer"
        JWT_AUTH[JWT Authentication]
        HEADER_EXTRACT[Header Extraction]
        UUID_VALIDATE[UUID Validation]
    end
    
    subgraph "Authorization Layer"
        TENANT_VALIDATE[Tenant Validation]
        ACCESS_CHECK[Access Control Check]
        CONTEXT_CREATE[Context Creation]
    end
    
    subgraph "Data Access Layer"
        COLLECTION_SCOPE[Collection Scoping]
        GRAPH_SCOPE[Graph Scoping]
        METADATA_FILTER[Metadata Filtering]
    end
    
    subgraph "Isolation Mechanisms"
        PHYSICAL[Physical Separation]
        LOGICAL[Logical Separation]
        FILTER_BASED[Filter-Based Isolation]
    end
    
    JWT_AUTH --> TENANT_VALIDATE
    HEADER_EXTRACT --> TENANT_VALIDATE
    UUID_VALIDATE --> ACCESS_CHECK
    
    TENANT_VALIDATE --> CONTEXT_CREATE
    ACCESS_CHECK --> CONTEXT_CREATE
    
    CONTEXT_CREATE --> COLLECTION_SCOPE
    CONTEXT_CREATE --> GRAPH_SCOPE
    CONTEXT_CREATE --> METADATA_FILTER
    
    COLLECTION_SCOPE --> PHYSICAL
    GRAPH_SCOPE --> PHYSICAL
    METADATA_FILTER --> LOGICAL
    METADATA_FILTER --> FILTER_BASED
```

---

## Technology Stack

### Technology Stack Overview

```mermaid
graph TB
    subgraph "Frontend/Client Layer"
        WEB_APPS[Web Applications]
        MOBILE_APPS[Mobile Applications]
        API_CLIENTS[API Clients]
    end
    
    subgraph "API Layer"
        FASTAPI[FastAPI 0.104+]
        PYDANTIC[Pydantic v2]
        UVICORN[Uvicorn ASGI Server]
    end
    
    subgraph "Authentication & Security"
        JWT_LIB[python-jose JWT]
        RS256[RS256 Algorithm]
        CORS_MW[CORS Middleware]
    end
    
    subgraph "Business Logic Layer"
        LANGCHAIN[LangChain Framework]
        LANGCHAIN_PG[LangChain PostgreSQL]
        LANGCHAIN_OPENAI[LangChain OpenAI]
        LANGCHAIN_COMMUNITY[LangChain Community]
    end
    
    subgraph "Document Processing"
        UNSTRUCTURED[Unstructured Library]
        AGENTIC_CHUNKER[Custom Agentic Chunker]
        GRAPH_TRANSFORMER[LLM Graph Transformer]
    end
    
    subgraph "Database Layer"
        POSTGRESQL[PostgreSQL 15+]
        PGVECTOR[pgvector Extension]
        APACHE_AGE[Apache AGE Extension]
        PSYCOPG2[psycopg2 Driver]
    end
    
    subgraph "AI/ML Services"
        OPENAI_API[OpenAI Compatible API]
        EMBEDDING_MODEL[text-embedding-3-large]
        LLM_MODELS[GPT-4o / Claude-3.5-Sonnet]
    end
    
    subgraph "Infrastructure"
        PYTHON[Python 3.9+]
        ASYNCIO[AsyncIO]
        BACKGROUND_TASKS[FastAPI Background Tasks]
        CONNECTION_POOL[Connection Pooling]
    end
    
    WEB_APPS --> FASTAPI
    MOBILE_APPS --> FASTAPI
    API_CLIENTS --> FASTAPI
    
    FASTAPI --> PYDANTIC
    FASTAPI --> UVICORN
    FASTAPI --> JWT_LIB
    FASTAPI --> CORS_MW
    
    FASTAPI --> LANGCHAIN
    LANGCHAIN --> LANGCHAIN_PG
    LANGCHAIN --> LANGCHAIN_OPENAI
    LANGCHAIN --> LANGCHAIN_COMMUNITY
    
    LANGCHAIN --> UNSTRUCTURED
    LANGCHAIN --> AGENTIC_CHUNKER
    LANGCHAIN --> GRAPH_TRANSFORMER
    
    LANGCHAIN_PG --> POSTGRESQL
    POSTGRESQL --> PGVECTOR
    POSTGRESQL --> APACHE_AGE
    POSTGRESQL --> PSYCOPG2
    
    LANGCHAIN_OPENAI --> OPENAI_API
    OPENAI_API --> EMBEDDING_MODEL
    OPENAI_API --> LLM_MODELS
    
    FASTAPI --> BACKGROUND_TASKS
    POSTGRESQL --> CONNECTION_POOL
```

### Technology Dependencies

```mermaid
graph LR
    subgraph "Core Dependencies"
        FASTAPI_DEP[fastapi>=0.104.0]
        PYDANTIC_DEP[pydantic>=2.0.0]
        UVICORN_DEP[uvicorn>=0.24.0]
    end
    
    subgraph "Database Dependencies"
        PSYCOPG2_DEP[psycopg2-binary>=2.9.0]
        LANGCHAIN_PG_DEP[langchain-postgres>=0.0.6]
    end
    
    subgraph "AI/ML Dependencies"
        LANGCHAIN_DEP[langchain>=0.1.0]
        LANGCHAIN_OPENAI_DEP[langchain-openai>=0.0.8]
        LANGCHAIN_COMMUNITY_DEP[langchain-community>=0.0.20]
        LANGCHAIN_EXPERIMENTAL_DEP[langchain-experimental>=0.0.50]
    end
    
    subgraph "Security Dependencies"
        JOSE_DEP[python-jose>=3.3.0]
        CRYPTOGRAPHY_DEP[cryptography>=41.0.0]
    end
    
    subgraph "Document Processing Dependencies"
        UNSTRUCTURED_DEP[unstructured>=0.10.0]
        PYDANTIC_SETTINGS_DEP[pydantic-settings>=2.0.0]
    end
    
    FASTAPI_DEP --> PYDANTIC_DEP
    FASTAPI_DEP --> UVICORN_DEP
    LANGCHAIN_DEP --> LANGCHAIN_OPENAI_DEP
    LANGCHAIN_DEP --> LANGCHAIN_COMMUNITY_DEP
    LANGCHAIN_DEP --> LANGCHAIN_EXPERIMENTAL_DEP
    JOSE_DEP --> CRYPTOGRAPHY_DEP
```

---

## Deployment Architecture

### Container Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end
    
    subgraph "Application Tier"
        subgraph "FastAPI Containers"
            APP1[FastAPI App 1]
            APP2[FastAPI App 2]
            APP3[FastAPI App N]
        end
        
        subgraph "Background Workers"
            WORKER1[Background Worker 1]
            WORKER2[Background Worker 2]
        end
    end
    
    subgraph "Database Tier"
        subgraph "PostgreSQL Cluster"
            PG_PRIMARY[PostgreSQL Primary]
            PG_REPLICA1[PostgreSQL Replica 1]
            PG_REPLICA2[PostgreSQL Replica 2]
        end
        
        subgraph "Extensions"
            PGVECTOR_EXT[pgvector Extension]
            AGE_EXT[Apache AGE Extension]
        end
    end
    
    subgraph "External Services"
        OPENAI_SVC[OpenAI Compatible API]
        MONITORING[Monitoring Stack]
        LOGGING[Centralized Logging]
    end
    
    LB --> APP1
    LB --> APP2
    LB --> APP3
    
    APP1 --> PG_PRIMARY
    APP2 --> PG_PRIMARY
    APP3 --> PG_PRIMARY
    
    APP1 -.-> PG_REPLICA1
    APP2 -.-> PG_REPLICA2
    
    WORKER1 --> PG_PRIMARY
    WORKER2 --> PG_PRIMARY
    
    PG_PRIMARY --> PGVECTOR_EXT
    PG_PRIMARY --> AGE_EXT
    
    APP1 --> OPENAI_SVC
    APP2 --> OPENAI_SVC
    APP3 --> OPENAI_SVC
    
    APP1 --> MONITORING
    APP2 --> MONITORING
    APP3 --> MONITORING
    
    APP1 --> LOGGING
    APP2 --> LOGGING
    APP3 --> LOGGING
```

### Kubernetes Deployment

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress"
            INGRESS[Ingress Controller]
            TLS[TLS Termination]
        end
        
        subgraph "Application Namespace"
            subgraph "FastAPI Deployment"
                FASTAPI_POD1[FastAPI Pod 1]
                FASTAPI_POD2[FastAPI Pod 2]
                FASTAPI_POD3[FastAPI Pod 3]
            end
            
            subgraph "Worker Deployment"
                WORKER_POD1[Worker Pod 1]
                WORKER_POD2[Worker Pod 2]
            end
            
            subgraph "Services"
                FASTAPI_SVC[FastAPI Service]
                WORKER_SVC[Worker Service]
            end
        end
        
        subgraph "Database Namespace"
            subgraph "PostgreSQL StatefulSet"
                PG_POD1[PostgreSQL Primary]
                PG_POD2[PostgreSQL Replica]
            end
            
            subgraph "Storage"
                PVC1[Persistent Volume 1]
                PVC2[Persistent Volume 2]
            end
        end
        
        subgraph "Configuration"
            CONFIG_MAP[ConfigMap]
            SECRETS[Secrets]
        end
    end
    
    INGRESS --> FASTAPI_SVC
    FASTAPI_SVC --> FASTAPI_POD1
    FASTAPI_SVC --> FASTAPI_POD2
    FASTAPI_SVC --> FASTAPI_POD3
    
    WORKER_SVC --> WORKER_POD1
    WORKER_SVC --> WORKER_POD2
    
    FASTAPI_POD1 --> PG_POD1
    FASTAPI_POD2 --> PG_POD1
    FASTAPI_POD3 --> PG_POD1
    
    WORKER_POD1 --> PG_POD1
    WORKER_POD2 --> PG_POD1
    
    PG_POD1 --> PVC1
    PG_POD2 --> PVC2
    
    FASTAPI_POD1 --> CONFIG_MAP
    FASTAPI_POD2 --> CONFIG_MAP
    FASTAPI_POD3 --> CONFIG_MAP
    
    FASTAPI_POD1 --> SECRETS
    FASTAPI_POD2 --> SECRETS
    FASTAPI_POD3 --> SECRETS
```

### Environment Configuration

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_API[FastAPI Dev Server]
        DEV_DB[Local PostgreSQL]
        DEV_CONFIG[.env Configuration]
    end
    
    subgraph "Staging Environment"
        STAGE_API[FastAPI Staging]
        STAGE_DB[Staging PostgreSQL]
        STAGE_CONFIG[Environment Variables]
    end
    
    subgraph "Production Environment"
        PROD_LB[Production Load Balancer]
        PROD_API[FastAPI Production Cluster]
        PROD_DB[Production PostgreSQL Cluster]
        PROD_CONFIG[Kubernetes Secrets]
    end
    
    DEV_API --> DEV_DB
    DEV_API --> DEV_CONFIG
    
    STAGE_API --> STAGE_DB
    STAGE_API --> STAGE_CONFIG
    
    PROD_LB --> PROD_API
    PROD_API --> PROD_DB
    PROD_API --> PROD_CONFIG
```

---

## Security Architecture

### Authentication & Authorization Flow

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant FastAPI
    participant JWT_Service
    participant TenantService
    participant Database
    
    Client->>Gateway: Request with JWT Token
    Gateway->>FastAPI: Forward Request
    FastAPI->>JWT_Service: Validate JWT Token
    JWT_Service->>JWT_Service: Verify RS256 Signature
    JWT_Service->>FastAPI: Return Token Payload
    
    FastAPI->>TenantService: Extract Tenant Context
    TenantService->>TenantService: Validate UUID Formats
    TenantService->>TenantService: Create Tenant Context
    TenantService->>FastAPI: Return Validated Context
    
    FastAPI->>Database: Execute Query with Tenant Filters
    Database->>FastAPI: Return Filtered Results
    FastAPI->>Client: Return Response
```

### Security Layers

```mermaid
graph TB
    subgraph "Network Security"
        TLS[TLS 1.3 Encryption]
        FIREWALL[Network Firewall]
        VPC[Virtual Private Cloud]
    end
    
    subgraph "Application Security"
        JWT_AUTH[JWT Authentication]
        CORS_POLICY[CORS Policy]
        RATE_LIMIT[Rate Limiting]
        INPUT_VALID[Input Validation]
    end
    
    subgraph "Data Security"
        TENANT_ISOLATION[Tenant Isolation]
        DATA_ENCRYPT[Data Encryption at Rest]
        CONN_ENCRYPT[Connection Encryption]
        ACCESS_CONTROL[Access Control]
    end
    
    subgraph "Infrastructure Security"
        CONTAINER_SEC[Container Security]
        SECRET_MGMT[Secret Management]
        AUDIT_LOG[Audit Logging]
        MONITORING[Security Monitoring]
    end
    
    TLS --> JWT_AUTH
    FIREWALL --> CORS_POLICY
    VPC --> RATE_LIMIT
    
    JWT_AUTH --> TENANT_ISOLATION
    INPUT_VALID --> DATA_ENCRYPT
    
    TENANT_ISOLATION --> CONTAINER_SEC
    ACCESS_CONTROL --> SECRET_MGMT
```

### Data Protection Model

```mermaid
graph TB
    subgraph "Data Classification"
        PUBLIC[Public Data]
        INTERNAL[Internal Data]
        CONFIDENTIAL[Confidential Data]
        RESTRICTED[Restricted Data]
    end
    
    subgraph "Protection Mechanisms"
        ENCRYPTION[AES-256 Encryption]
        TOKENIZATION[Data Tokenization]
        MASKING[Data Masking]
        ANONYMIZATION[Data Anonymization]
    end
    
    subgraph "Access Controls"
        RBAC[Role-Based Access Control]
        ABAC[Attribute-Based Access Control]
        TENANT_FILTER[Tenant-Based Filtering]
        API_KEYS[API Key Management]
    end
    
    subgraph "Compliance"
        GDPR[GDPR Compliance]
        SOC2[SOC 2 Compliance]
        AUDIT_TRAIL[Audit Trail]
        DATA_RETENTION[Data Retention Policies]
    end
    
    CONFIDENTIAL --> ENCRYPTION
    RESTRICTED --> TOKENIZATION
    INTERNAL --> MASKING
    
    ENCRYPTION --> RBAC
    TOKENIZATION --> ABAC
    MASKING --> TENANT_FILTER
    
    RBAC --> GDPR
    TENANT_FILTER --> SOC2
    API_KEYS --> AUDIT_TRAIL
```

---

## Performance Considerations

### Performance Architecture

```mermaid
graph TB
    subgraph "Caching Strategy"
        REDIS[Redis Cache]
        APP_CACHE[Application Cache]
        DB_CACHE[Database Query Cache]
        EMBEDDING_CACHE[Embedding Cache]
    end
    
    subgraph "Database Optimization"
        CONN_POOL[Connection Pooling]
        READ_REPLICA[Read Replicas]
        INDEX_OPT[Index Optimization]
        QUERY_OPT[Query Optimization]
    end
    
    subgraph "Application Optimization"
        ASYNC_PROC[Async Processing]
        BACKGROUND_TASKS[Background Tasks]
        BATCH_PROC[Batch Processing]
        LAZY_LOAD[Lazy Loading]
    end
    
    subgraph "Infrastructure Optimization"
        LOAD_BALANCE[Load Balancing]
        AUTO_SCALE[Auto Scaling]
        CDN[Content Delivery Network]
        MONITORING[Performance Monitoring]
    end
    
    REDIS --> CONN_POOL
    APP_CACHE --> READ_REPLICA
    DB_CACHE --> INDEX_OPT
    EMBEDDING_CACHE --> QUERY_OPT
    
    CONN_POOL --> ASYNC_PROC
    READ_REPLICA --> BACKGROUND_TASKS
    INDEX_OPT --> BATCH_PROC
    
    ASYNC_PROC --> LOAD_BALANCE
    BACKGROUND_TASKS --> AUTO_SCALE
    BATCH_PROC --> CDN
```

### Scalability Patterns

```mermaid
graph TB
    subgraph "Horizontal Scaling"
        API_SCALE[API Server Scaling]
        WORKER_SCALE[Background Worker Scaling]
        DB_SHARD[Database Sharding]
        TENANT_SHARD[Tenant-Based Sharding]
    end
    
    subgraph "Vertical Scaling"
        CPU_SCALE[CPU Scaling]
        MEMORY_SCALE[Memory Scaling]
        STORAGE_SCALE[Storage Scaling]
        NETWORK_SCALE[Network Scaling]
    end
    
    subgraph "Performance Metrics"
        LATENCY[Response Latency]
        THROUGHPUT[Request Throughput]
        CONCURRENCY[Concurrent Users]
        RESOURCE_UTIL[Resource Utilization]
    end
    
    subgraph "Optimization Strategies"
        CACHING[Intelligent Caching]
        PREFETCH[Data Prefetching]
        COMPRESSION[Data Compression]
        PAGINATION[Result Pagination]
    end
    
    API_SCALE --> LATENCY
    WORKER_SCALE --> THROUGHPUT
    DB_SHARD --> CONCURRENCY
    TENANT_SHARD --> RESOURCE_UTIL
    
    LATENCY --> CACHING
    THROUGHPUT --> PREFETCH
    CONCURRENCY --> COMPRESSION
    RESOURCE_UTIL --> PAGINATION
```

---

## System Interactions and Detailed Explanations

### Document Ingestion Workflow Explanation

The document ingestion process is a sophisticated pipeline that transforms raw documents into searchable knowledge:

1. **Document Upload**: Users upload documents through the REST API with optional metadata
2. **Content Extraction**: The Unstructured library parses various file formats (PDF, DOCX, TXT, etc.)
3. **Proposition Generation**: An LLM extracts atomic propositions from the document content
4. **Agentic Chunking**: The custom AgenticChunker intelligently groups related propositions
5. **Vector Storage**: Chunks are embedded and stored in pgvector with tenant metadata
6. **Graph Extraction**: LLMGraphTransformer extracts entities and relationships
7. **Graph Storage**: Knowledge graph elements are stored in Apache AGE with tenant isolation

### Multi-Tenant Data Isolation

The system implements a hierarchical multi-tenant architecture:

- **Client Level**: Top-level tenant isolation with separate vector collections and graph instances
- **Project Level**: Sub-tenant isolation within client boundaries
- **Workspace Level**: Fine-grained isolation for team-based access

Each level generates unique collection and graph names using a consistent naming convention that ensures complete data separation while maintaining efficient access patterns.

### Dual Search Capabilities

The system provides two complementary search approaches:

**Vector Search**: 
- Uses semantic similarity through embeddings
- Excellent for finding conceptually related content
- Leverages pgvector for efficient similarity calculations
- Returns contextually relevant document chunks

**Graph Search**:
- Uses knowledge graph relationships
- Excellent for finding connected entities and concepts
- Leverages Apache AGE for complex graph traversals
- Returns structured relationship information

### Security and Authentication

The security model implements multiple layers of protection:

- **JWT Authentication**: RS256 algorithm with public key verification
- **Tenant Validation**: UUID format validation and access control
- **Data Isolation**: Physical and logical separation of tenant data
- **Input Validation**: Comprehensive validation using Pydantic models
- **Connection Security**: Encrypted database connections and secure API endpoints

### Performance and Scalability

The architecture is designed for enterprise-scale performance:

- **Async Processing**: Non-blocking I/O for high concurrency
- **Background Tasks**: Resource-intensive operations run asynchronously
- **Connection Pooling**: Efficient database connection management
- **Caching Strategy**: Multi-level caching for frequently accessed data
- **Horizontal Scaling**: Stateless design enables easy scaling

---

## Conclusion

This FastAPI RAG application represents a sophisticated enterprise-level solution that combines the power of vector similarity search with knowledge graph capabilities. The multi-tenant architecture ensures secure data isolation while maintaining high performance and scalability.

Key architectural strengths include:

- **Modular Design**: Clear separation of concerns enables easy maintenance and extension
- **Dual Search Capabilities**: Vector and graph search provide comprehensive retrieval options
- **Enterprise Security**: Robust authentication and tenant isolation mechanisms
- **Intelligent Processing**: Agentic chunking and graph extraction optimize knowledge representation
- **Scalable Infrastructure**: Designed for horizontal scaling and high availability

The system is well-positioned to handle enterprise workloads while providing the flexibility to adapt to evolving requirements and integrate with existing enterprise systems.

---

## Appendix

### API Endpoint Reference

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/` | GET | Root endpoint with application info | No |
| `/health` | GET | Health check endpoint | No |
| `/tenant/validate` | GET | Validate tenant context | Yes |
| `/tenant/info` | GET | Get tenant information | Yes |
| `/ingest/` | POST | Ingest document | Yes |
| `/ingest/status/{task_id}` | GET | Get ingestion status | Yes |
| `/search/vector-search` | POST | Vector similarity search | Yes |
| `/search/graph-search` | POST | Graph-based search | Yes |

### Configuration Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `POSTGRES_HOST` | PostgreSQL host | localhost | Yes |
| `POSTGRES_PORT` | PostgreSQL port | 5432 | Yes |
| `POSTGRES_DB` | Database name | fast_rag_db | Yes |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `JWT_PUBLIC_KEY` | JWT public key | - | Yes |
| `EMBEDDING_MODEL_NAME` | Embedding model | text-embedding-3-large | No |
| `VECTOR_SEARCH_TOP_K` | Vector search results | 10 | No |
| `ENABLE_TENANT_ISOLATION` | Enable multi-tenancy | False | No |

### Database Extensions Required

- **pgvector**: Vector similarity search capabilities
- **Apache AGE**: Graph database functionality

Both extensions must be installed and configured in the PostgreSQL instance before running the application.
