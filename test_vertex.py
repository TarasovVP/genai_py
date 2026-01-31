from google import genai

client = genai.Client(vertexai=True, project="your-gcp-project", location="europe-west1")

resp = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Say hello in one sentence."
)
print(resp.text)