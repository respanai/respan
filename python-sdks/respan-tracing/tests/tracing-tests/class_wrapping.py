from respan_tracing.contexts.span import respan_span_attributes
from openai import OpenAI
from anthropic import Anthropic
from respan_tracing import RespanTelemetry
from respan_tracing.decorators import workflow, task
from dotenv import load_dotenv
import time

load_dotenv(override=True)

k_tl = RespanTelemetry()
client = OpenAI()
anthropic = Anthropic()

@workflow(name="joke_agent", method_name="run")
class JokeAgent:
    def __init__(self):
        self.client = OpenAI()
        self.anthropic = Anthropic()

    @task(name="store_joke")
    def store_joke(self, joke: str):
        embedding = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=joke,
        )
        return embedding.data[0].embedding

    @task(name="joke_creation")
    def create_joke(self):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
            temperature=0.5,
            max_tokens=100,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["\n"],
            logprobs=True,
        )
        joke = completion.choices[0].message.content
        self.store_joke(joke)
        return joke

    @task(name="translate_joke_to_pirate")
    def translate_to_pirate(self, joke: str):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "translate the joke to pirate language:\n\n" + joke}],
        )
        return completion.choices[0].message.content

    @task(name="audience_laughs")
    def audience_laughs(self, joke: str):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "This joke:\n\n" + joke + " is funny, say hahahahaha"}],
            max_tokens=10,
        )
        return completion.choices[0].message.content

    @task(name="audience_claps")
    def audience_claps(self):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Clap once"}],
            max_tokens=5,
        )
        return completion.choices[0].message.content

    @task(name="audience_applaud")
    def audience_applaud(self, joke: str):
        clap = self.audience_claps()
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Applaud to the joke, clap clap! " + clap}],
            max_tokens=10,
        )
        return completion.choices[0].message.content

    @task(name="ask_for_comments")
    def ask_for_comments(self, joke: str):
        completion = self.anthropic.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": f"What do you think about this joke: {joke}"}],
            max_tokens=100,
        )
        return completion.content[0].text

    @task(name="logging_joke")
    def log_joke(self, joke: str, reactions: str):
        print(joke + "\n\n" + reactions)
        time.sleep(1)

    def run(self):
        # Main workflow
        joke = self.create_joke()
        pirate_joke = self.translate_to_pirate(joke)
        
        # Audience reactions
        laughs = self.audience_laughs(pirate_joke)
        applauds = self.audience_applaud(pirate_joke)
        reactions = laughs + applauds
        
        # Get comments
        comments = self.ask_for_comments(pirate_joke)
        
        # Log everything
        self.log_joke(pirate_joke, reactions + "\n" + comments)
        
        return {
            "joke": pirate_joke,
            "reactions": reactions,
            "comments": comments
        }

if __name__ == "__main__":
    agent = JokeAgent()
    result = agent.run()
    print("\nFinal Result:", result)

