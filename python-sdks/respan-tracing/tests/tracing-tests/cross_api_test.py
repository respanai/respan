from dotenv import load_dotenv
load_dotenv("tests/.env", override=True)

"""
Call an endpoint and check if the telemetry headers are included in the request
"""

from respan_tracing import RespanTelemetry, workflow
from requests import post

k_tl = RespanTelemetry(
)


topics = ["black hole"]
prompt_template = """
You're an expert science communicator, able to explain complex topics in an
approachable manner. Your task is to respond to the questions of users in an
engaging, informative, and friendly way. Stay factual, and refrain from using
jargon. Your answer should be 4 sentences at max.
Remember, keep it ENGAGING and FUN!
 
Question: {question}
"""


@workflow(name="send_webhook")
def send_webhook(topic):
    prompt = prompt_template.format(question=topic)

    response = post(
        url="https://webhook.site/5d2f431e-1ed6-4639-bd6c-dc19de391a50",
        json={"prompt": prompt},
    )

    return response.text


for topic in topics:
    print(f"Input: Please explain to me {topic.lower()}")
    print(f"Answer: {send_webhook(topic)} \n")
