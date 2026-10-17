import os

from google import genai

project = os.environ["VERTEX_PROJECT"]

client = genai.Client(vertexai=True, project=project, location="europe-west1")

resp = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Say hello in one sentence."
)
print(resp.text)
