import requests

API_URL = "http://localhost:8000"

questions = [
    "What is the refund policy?",
    "How much does the Pro plan cost?",
    "What devices are compatible?",
    "How do I cook pasta?",
    "What's the weather today?"
]

if __name__ == "__main__":
    for q in questions:
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"query": q, "temperature": 0.7}
            )
            result = response.json()
            
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            print(f"Allowed: {result['allowed']}")
            print(f"Score: {result['score']:.3f}")
            
            if result['allowed']:
                print(f"Answer: {result['answer'][:100]}...")
            else:
                print("Answer: REFUSED (outside knowledge boundary)")
            
            print(f"Latency: {result['latency_ms']:.1f}ms")
        except Exception as e:
            print(f"Could not connect to {API_URL}: {e}")

