#!/bin/bash
# Start Qdrant vector database locally via Docker

docker run -d \
  --name qdrant-student-records \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant:latest

echo "Qdrant started at http://localhost:6333"
echo "Dashboard: http://localhost:6333/dashboard"
