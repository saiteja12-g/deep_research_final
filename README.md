```
docker run `
    -p 7474:7474 `
    -p 7687:7687 `
    -v ${PWD}/neo4j/data:/data `
    -v ${PWD}/neo4j/import:/import `
    --env NEO4J_AUTH=neo4j/research123 `
    neo4j:latest
```