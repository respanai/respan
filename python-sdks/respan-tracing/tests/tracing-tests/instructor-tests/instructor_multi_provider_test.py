"""
Multi-provider test demonstrating Respan tracing with Instructor across different LLM providers.

This test shows how Respan tracing works with:
- OpenAI GPT models
- Anthropic Claude models
- Different model capabilities and structured outputs
- Provider-specific features and parameters
"""

from dotenv import load_dotenv
import os
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import instructor
from respan_tracing import RespanTelemetry
from respan_tracing.decorators import task, workflow

# Load environment variables
load_dotenv(".env", override=True)

# Initialize Respan Telemetry
k_tl = RespanTelemetry(app_name="instructor-multi-provider-test")

# Define shared Pydantic models
class Sentiment(str, Literal["positive", "negative", "neutral"]):
    pass

class TextAnalysis(BaseModel):
    """Analysis of a text with sentiment and key points."""
    text: str = Field(description="The original text being analyzed")
    sentiment: Sentiment = Field(description="Overall sentiment of the text")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    key_points: List[str] = Field(description="Key points or themes", min_length=1, max_length=5)
    summary: str = Field(description="Brief summary of the text", max_length=200)

class CodeExplanation(BaseModel):
    """Explanation of code functionality."""
    language: str = Field(description="Programming language")
    purpose: str = Field(description="What the code does")
    complexity: Literal["beginner", "intermediate", "advanced"] = Field(description="Complexity level")
    key_concepts: List[str] = Field(description="Key programming concepts used")
    explanation: str = Field(description="Detailed explanation of the code")

class CreativeStory(BaseModel):
    """A creative story with structured elements."""
    title: str = Field(description="Story title")
    genre: str = Field(description="Story genre")
    main_character: str = Field(description="Main character name")
    setting: str = Field(description="Where the story takes place")
    plot_summary: str = Field(description="Brief plot summary")
    story: str = Field(description="The full story", min_length=100)

# Initialize clients for different providers
openai_client = instructor.from_provider("openai/gpt-4o-mini")

# Check if Anthropic is available
try:
    anthropic_client = instructor.from_provider("anthropic/claude-haiku-4-5-20251001")
    ANTHROPIC_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Anthropic not available: {e}")
    anthropic_client = None
    ANTHROPIC_AVAILABLE = False

@task(name="openai_text_analysis")
def analyze_text_with_openai(text: str) -> TextAnalysis:
    """Analyze text sentiment using OpenAI."""
    analysis = openai_client.chat.completions.create(
        response_model=TextAnalysis,
        messages=[
            {"role": "system", "content": "Analyze the sentiment and key points of the provided text. Be thorough and accurate."},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        max_tokens=600
    )
    return analysis

@task(name="anthropic_text_analysis")
def analyze_text_with_anthropic(text: str) -> Optional[TextAnalysis]:
    """Analyze text sentiment using Anthropic Claude."""
    if not ANTHROPIC_AVAILABLE:
        print("⚠️ Skipping Anthropic test - not available")
        return None
    
    analysis = anthropic_client.chat.completions.create(
        response_model=TextAnalysis,
        messages=[
            {"role": "system", "content": "Analyze the sentiment and key points of the provided text. Be thorough and accurate."},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        max_tokens=600
    )
    return analysis

@task(name="openai_code_explanation")
def explain_code_with_openai(code: str) -> CodeExplanation:
    """Explain code using OpenAI."""
    explanation = openai_client.chat.completions.create(
        response_model=CodeExplanation,
        messages=[
            {"role": "system", "content": "Explain the provided code in detail. Identify the language, purpose, complexity, and key concepts."},
            {"role": "user", "content": f"Explain this code:\n\n{code}"}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return explanation

@task(name="anthropic_code_explanation")
def explain_code_with_anthropic(code: str) -> Optional[CodeExplanation]:
    """Explain code using Anthropic Claude."""
    if not ANTHROPIC_AVAILABLE:
        print("⚠️ Skipping Anthropic test - not available")
        return None
    
    explanation = anthropic_client.chat.completions.create(
        response_model=CodeExplanation,
        messages=[
            {"role": "system", "content": "Explain the provided code in detail. Identify the language, purpose, complexity, and key concepts."},
            {"role": "user", "content": f"Explain this code:\n\n{code}"}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return explanation

@task(name="openai_creative_writing")
def create_story_with_openai(prompt: str) -> CreativeStory:
    """Create a creative story using OpenAI."""
    story = openai_client.chat.completions.create(
        response_model=CreativeStory,
        messages=[
            {"role": "system", "content": "Create an engaging short story based on the prompt. Include all required story elements."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # Higher temperature for creativity
        max_tokens=1000
    )
    return story

@task(name="anthropic_creative_writing")
def create_story_with_anthropic(prompt: str) -> Optional[CreativeStory]:
    """Create a creative story using Anthropic Claude."""
    if not ANTHROPIC_AVAILABLE:
        print("⚠️ Skipping Anthropic test - not available")
        return None
    
    story = anthropic_client.chat.completions.create(
        response_model=CreativeStory,
        messages=[
            {"role": "system", "content": "Create an engaging short story based on the prompt. Include all required story elements."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # Higher temperature for creativity
        max_tokens=1000
    )
    return story

@workflow(name="multi_provider_comparison")
def run_multi_provider_tests():
    """Run the same tasks across different providers for comparison."""
    
    # Test 1: Text Analysis Comparison
    print("\n=== Test 1: Text Analysis Comparison ===")
    sample_text = """
    I absolutely love the new features in this software update! The user interface is so much more 
    intuitive and the performance improvements are incredible. However, I'm a bit concerned about 
    the learning curve for some of the advanced features. Overall, this is a fantastic update that 
    will really help our team be more productive.
    """
    
    print("📝 Analyzing text with OpenAI...")
    openai_analysis = analyze_text_with_openai(sample_text)
    print(f"   OpenAI Sentiment: {openai_analysis.sentiment} (confidence: {openai_analysis.confidence:.2f})")
    print(f"   OpenAI Key Points: {', '.join(openai_analysis.key_points)}")
    
    anthropic_analysis = None
    if ANTHROPIC_AVAILABLE:
        print("📝 Analyzing text with Anthropic...")
        anthropic_analysis = analyze_text_with_anthropic(sample_text)
        print(f"   Anthropic Sentiment: {anthropic_analysis.sentiment} (confidence: {anthropic_analysis.confidence:.2f})")
        print(f"   Anthropic Key Points: {', '.join(anthropic_analysis.key_points)}")
    
    # Test 2: Code Explanation Comparison
    print("\n=== Test 2: Code Explanation Comparison ===")
    sample_code = """
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    
    # Generate first 10 Fibonacci numbers
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")
    """
    
    print("💻 Explaining code with OpenAI...")
    openai_code = explain_code_with_openai(sample_code)
    print(f"   OpenAI Language: {openai_code.language}")
    print(f"   OpenAI Complexity: {openai_code.complexity}")
    print(f"   OpenAI Purpose: {openai_code.purpose}")
    
    anthropic_code = None
    if ANTHROPIC_AVAILABLE:
        print("💻 Explaining code with Anthropic...")
        anthropic_code = explain_code_with_anthropic(sample_code)
        print(f"   Anthropic Language: {anthropic_code.language}")
        print(f"   Anthropic Complexity: {anthropic_code.complexity}")
        print(f"   Anthropic Purpose: {anthropic_code.purpose}")
    
    # Test 3: Creative Writing Comparison
    print("\n=== Test 3: Creative Writing Comparison ===")
    story_prompt = "Write a short story about a robot who discovers they can dream."
    
    print("✨ Creating story with OpenAI...")
    openai_story = create_story_with_openai(story_prompt)
    print(f"   OpenAI Story: '{openai_story.title}' ({openai_story.genre})")
    print(f"   OpenAI Character: {openai_story.main_character}")
    print(f"   OpenAI Setting: {openai_story.setting}")
    
    anthropic_story = None
    if ANTHROPIC_AVAILABLE:
        print("✨ Creating story with Anthropic...")
        anthropic_story = create_story_with_anthropic(story_prompt)
        print(f"   Anthropic Story: '{anthropic_story.title}' ({anthropic_story.genre})")
        print(f"   Anthropic Character: {anthropic_story.main_character}")
        print(f"   Anthropic Setting: {anthropic_story.setting}")
    
    return {
        "text_analysis": {
            "openai": openai_analysis,
            "anthropic": anthropic_analysis
        },
        "code_explanation": {
            "openai": openai_code,
            "anthropic": anthropic_code
        },
        "creative_writing": {
            "openai": openai_story,
            "anthropic": anthropic_story
        }
    }

@task(name="compare_provider_results")
def compare_provider_results(results: dict):
    """Compare results from different providers."""
    print("\n=== Provider Comparison Summary ===")
    
    # Compare text analysis
    openai_analysis = results["text_analysis"]["openai"]
    anthropic_analysis = results["text_analysis"]["anthropic"]
    
    print(f"📊 Text Analysis:")
    print(f"   OpenAI Sentiment: {openai_analysis.sentiment}")
    if anthropic_analysis:
        print(f"   Anthropic Sentiment: {anthropic_analysis.sentiment}")
        sentiment_match = openai_analysis.sentiment == anthropic_analysis.sentiment
        print(f"   Sentiment Agreement: {'✅' if sentiment_match else '❌'}")
    
    # Compare code explanations
    openai_code = results["code_explanation"]["openai"]
    anthropic_code = results["code_explanation"]["anthropic"]
    
    print(f"\n💻 Code Explanation:")
    print(f"   OpenAI Language: {openai_code.language}")
    if anthropic_code:
        print(f"   Anthropic Language: {anthropic_code.language}")
        language_match = openai_code.language.lower() == anthropic_code.language.lower()
        print(f"   Language Agreement: {'✅' if language_match else '❌'}")
    
    # Compare creative outputs
    openai_story = results["creative_writing"]["openai"]
    anthropic_story = results["creative_writing"]["anthropic"]
    
    print(f"\n✨ Creative Writing:")
    print(f"   OpenAI Genre: {openai_story.genre}")
    if anthropic_story:
        print(f"   Anthropic Genre: {anthropic_story.genre}")
    
    print(f"\n🎯 Provider Availability:")
    print(f"   OpenAI: ✅ Available")
    print(f"   Anthropic: {'✅ Available' if ANTHROPIC_AVAILABLE else '❌ Not Available'}")
    
    return True

@workflow(name="instructor_multi_provider_demo")
def main():
    """Main workflow demonstrating Instructor with multiple providers."""
    print("🚀 Starting Multi-Provider Instructor + Respan Tracing Demo")
    print("=" * 65)
    
    # Check provider availability
    print(f"🔍 Provider Status:")
    print(f"   OpenAI: ✅ Available")
    print(f"   Anthropic: {'✅ Available' if ANTHROPIC_AVAILABLE else '❌ Not Available (check API key)'}")
    
    if not ANTHROPIC_AVAILABLE:
        print("\n⚠️ Note: Some tests will be skipped due to missing Anthropic configuration")
    
    # Run multi-provider tests
    results = run_multi_provider_tests()
    
    # Compare results
    comparison_done = compare_provider_results(results)
    
    print("\n" + "=" * 65)
    print("✅ Multi-provider demo completed successfully!")
    print(f"✅ Comparison completed: {comparison_done}")
    print("\n🌟 Features demonstrated:")
    print("   ✅ OpenAI GPT-4o-mini structured outputs")
    if ANTHROPIC_AVAILABLE:
        print("   ✅ Anthropic Claude structured outputs")
    else:
        print("   ⚠️ Anthropic Claude (skipped - not configured)")
    print("   ✅ Provider-specific parameter handling")
    print("   ✅ Cross-provider result comparison")
    print("   ✅ Unified tracing across providers")
    print("\n📈 Check your Respan dashboard for:")
    print("   - Separate traces for each provider")
    print("   - Token usage comparison")
    print("   - Response time differences")
    print("   - Model-specific parameters")
    print("   - Provider-specific metadata")
    
    return results

if __name__ == "__main__":
    main()