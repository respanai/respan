from dotenv import load_dotenv
load_dotenv("tests/.env", override=True)

import openai
from respan_tracing import RespanTelemetry, workflow, task
k_tl = RespanTelemetry()


topics = ["black hole"]
prompt_template = """
You're an expert science communicator, able to explain complex topics in an
approachable manner. Your task is to respond to the questions of users in an
engaging, informative, and friendly way. Stay factual, and refrain from using
jargon. Your answer should be 4 sentences at max.
Remember, keep it ENGAGING and FUN!
 
Question: {question}
"""
from time import sleep
from threading import Thread


@task(name="another_generation")
def another_generation(prompt):
    sleep(10)
    return "This is the response of other generation"


@workflow(name="explain_concept")
def explain_concept(topic):
    prompt = prompt_template.format(question=topic)
    thread = Thread(target=another_generation, args=(prompt,))
    thread.start()
    # another_generation(prompt)
    response = (
        openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o-mini",
            temperature=0.6,
        )
        .choices[0]
        .message.content
    )

    # IMPORTANT FIX: Wait for the background thread to complete
    # This ensures the span gets properly closed and exported
    thread.join()

    return response

def run_tracing_test():
    for topic in topics:
        print(f"Input: Please explain to me {topic.lower()}")
        print(f"Answer: {explain_concept(topic)} \n")
