import anthropic
import os


def optimize_prompt(user_input: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"""You are an expert at writing prompts for Stable Diffusion XL image generation.

Convert the following user input into an optimized English prompt for SDXL.
Rules:
- Output ONLY the prompt, no explanation
- Use descriptive English keywords
- Add quality boosters: high quality, detailed, 8k, photorealistic (when appropriate)
- Add style/lighting descriptors
- Keep it under 77 tokens
- If input is Korean, translate and expand it

User input: {user_input}

Optimized prompt:""",
            }
        ],
    )

    return message.content[0].text.strip()
