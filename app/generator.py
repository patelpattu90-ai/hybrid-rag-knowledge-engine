from groq import Groq


class GroqGenerator:

    def __init__(self, api_key):

        self.client = Groq(api_key=api_key)

        self.model = "llama-3.3-70b-versatile"


    def generate(self, query, contexts):

        context_text = "\n\n".join([c["text"] for c in contexts])

        prompt = f"""
You are a technical documentation assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the documentation."

Context:
{context_text}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content