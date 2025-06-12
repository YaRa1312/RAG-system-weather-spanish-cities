# Weather Data Retrieval and Query System

The project implementing a system to fetch, store, and query weather data for cities using a vector database for semantic search.

# Technologies / Tools used

```
Python
Pinecone
SentenceTransformers
```

# Key Decisions

**Embedding Model (all-MiniLM-L6-v2)**: Chosen for its compact size and efficiency in generating 384-dimensional embeddings, suitable for semantic text similarity tasks. It balances performance and resource usage, making it a good choice for a small-scale application.

**Pinecone Index**: Used a serverless Pinecone index with cosine similarity metric for efficient vector storage and retrieval. The dimension (384) matches the embedding model output.

**Cosine Similarity Threshold (0.3)**: Set a low threshold to avoid overly strict matching, allowing flexibility in user queries but with a risk of less precise matches.

**City Data Storage**: Stored weather data (temperature, wind speed) as metadata in Pinecone, linked to city name embeddings, enabling fast retrieval based on semantic queries.

# Limitations

**Limited City Data**: The system is hardcoded to three cities (Madrid, Barcelona, Bilbao). Expanding to a larger, dynamic city dataset would improve usability.

**API Dependency**: Relies on an external weather API, which could fail or rate-limit. Caching weather data or using a fallback API could enhance reliability.

**Similarity Threshold**: The 0.3 threshold may lead to false positives for ambiguous queries. Fine-tuning the threshold or implementing a secondary validation step could improve accuracy.

**Query Flexibility**: The system assumes city names are explicit in queries. Adding natural language processing to extract city names from complex queries would make it more robust.

**Scalability**: For large-scale use, batching upserts to Pinecone and optimizing embedding generation would improve performance.

# Potential Improvements

> Integrate a geocoding API to fetch coordinates for any city dynamically.

> Add support for more weather parameters (e.g., humidity, precipitation).

> Implement a user interface for interactive querying.