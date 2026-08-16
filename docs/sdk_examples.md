# MYAI SDK Examples

## curl

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the refund policy?",
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Python

```python
import requests

API_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Ask a question
response = requests.post(
    f"{API_URL}/ask",
    json={
        "query": "How much does the Pro plan cost?",
        "temperature": 0.7,
        "max_tokens": 256
    }
)

result = response.json()
if result["allowed"]:
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']:.3f}")
else:
    print("Question outside knowledge boundary")
```

## JavaScript (Node.js)

```javascript
const API_URL = "http://localhost:8000";

async function ask(query) {
  const response = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: query,
      temperature: 0.7,
      max_tokens: 256
    })
  });
  
  const result = await response.json();
  
  if (result.allowed) {
    console.log(`Answer: ${result.answer}`);
    console.log(`Score: ${result.score.toFixed(3)}`);
    console.log(`Latency: ${result.latency_ms.toFixed(1)}ms`);
  } else {
    console.log("Question outside knowledge boundary");
  }
}

ask("What devices are compatible?");
```

## JavaScript (Browser)

```javascript
async function askQuestion(query) {
  const response = await fetch("http://localhost:8000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: query,
      temperature: 0.7,
      max_tokens: 256
    })
  });
  
  return await response.json();
}

// Usage
const result = await askQuestion("What is the refund policy?");
if (result.allowed) {
  document.getElementById("answer").textContent = result.answer;
}
```
