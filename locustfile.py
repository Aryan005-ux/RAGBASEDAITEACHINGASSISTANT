import random
from locust import HttpUser, task, between, events

# A pool of realistic Sigma Course queries for testing
QUERIES = [
    "How to install VS Code?",
    "How websites work?",
    "What is the basic structure of HTML?",
    "Explain heading, paragraphs, and links",
    "How to add images, lists, and tables in HTML?",
    "What is SEO and Core Web Vitals?",
    "How do forms and inputs work in HTML?",
    "What is the difference between Inline & Block elements?",
    "How to use Id & Classes in HTML?",
    "How to insert video, audio, and media in HTML?",
    "What are HTML Semantic Tags?",
    "HTML entities and code tags",
    "Introduction to CSS",
    "Difference between inline, internal, and external CSS",
    "How to use CSS Selectors?",
    "What is the CSS Box Model?",
    "How to preview website on mobile?",
    "What is the chat GPT impact on coding?"
]

class SigmaRAGUser(HttpUser):
    # Wait between 0.9 and 1.1 seconds between tasks.
    # With 200 concurrent users, this simulates a steady target workload of 200 req/s.
    wait_time = between(0.9, 1.1)

    @task(70)
    def search_endpoint(self):
        """Simulate users searching for video chunks."""
        query = random.choice(QUERIES)
        payload = {
            "query": query,
            "top_k": random.randint(3, 7)
        }
        
        with self.client.post("/search", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                # Validate headers
                latency = response.headers.get("X-Inference-Latency-MS")
                cost_lambda = response.headers.get("X-Inference-Cost-USD-Lambda")
                cost_ecs = response.headers.get("X-Inference-Cost-USD-ECS")
                sla_status = response.headers.get("X-SLA-Compliance")
                
                if latency and cost_lambda and cost_ecs:
                    response.success()
                else:
                    response.failure("Missing SLA/Cost headers in response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(25)
    def query_endpoint(self):
        """Simulate users asking questions to the assistant (RAG query)."""
        query = random.choice(QUERIES)
        payload = {
            "query": query
        }
        
        with self.client.post("/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                # Validate response JSON structure
                resp_json = response.json()
                if "answer" in resp_json and "context_chunks" in resp_json:
                    response.success()
                else:
                    response.failure("Response missing 'answer' or 'context_chunks'")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(5)
    def health_check(self):
        """Simulate ALB health check polls."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200 and response.json().get("status") == "healthy":
                response.success()
            else:
                response.failure("Health check failed")

# Event hooks for custom metric logging in console (helpful for headless Docker tests)
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, exception, **kwargs):
    if exception:
        print(f"Request FAILED: {request_type} {name} - Latency: {response_time}ms - Error: {exception}")
