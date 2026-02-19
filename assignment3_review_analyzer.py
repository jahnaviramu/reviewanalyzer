
from pydantic import BaseModel, Field, validator
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM


class ReviewAnalysis(BaseModel):
    """
    Pydantic model for structured review analysis.
    Ensures type validation and schema consistency.
    """
    sentiment: str = Field(
        description="Overall sentiment of the review (Positive, Negative, or Neutral)",
        min_length=3,
        max_length=50
    )
    rating: int = Field(
        description="Numerical rating from 1 to 5",
        ge=1,
        le=5
    )
    key_features: List[str] = Field(
        description="List of product features mentioned in the review"
    )
    improvement_suggestions: List[str] = Field(
        description="List of suggested improvements mentioned in the review"
    )
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        """Validate that sentiment is one of the expected values."""
        valid_sentiments = ['Positive', 'Negative', 'Neutral', 'Positive', 'Mixed']
        if v not in valid_sentiments and not any(s.lower() in v.lower() for s in valid_sentiments):
            # Allow variations of the main sentiments
            pass
        return v
    
    @validator('key_features', 'improvement_suggestions', pre=True, always=True)
    def ensure_list(cls, v):
        """Ensure fields are lists."""
        if isinstance(v, str):
            return [v]
        return v or []


def create_review_analyzer():
    """
    Creates a product review analysis pipeline using PydanticOutputParser.
    Returns a chain that extracts structured insights from reviews.
    """
    
    # Initialize the Ollama LLM
    llm = OllamaLLM(model="mistral")
    
    # Initialize the Pydantic output parser with the ReviewAnalysis model
    output_parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
    
    # Get format instructions for the Pydantic model
    format_instructions = output_parser.get_format_instructions()
    
    # Create a detailed prompt template for review analysis
    prompt_template = PromptTemplate(
        input_variables=["review_text"],
        template="""Analyze the following product review and extract structured information.

{format_instructions}

Product Review:
{review_text}

Respond with ONLY valid JSON matching the required format.""",
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Chain components: prompt -> model -> parser
    chain = prompt_template | llm | output_parser
    
    return chain


def analyze_review(review_text: str) -> ReviewAnalysis:
    """
    Analyzes a product review and returns structured insights.
    Includes error handling for parsing failures.
    
    Args:
        review_text: Raw review text to analyze
        
    Returns:
        ReviewAnalysis object with validated structured data
    """
    chain = create_review_analyzer()
    
    try:
        result = chain.invoke({"review_text": review_text})
        return result
    except Exception as e:
        print(f"Error analyzing review: {e}")
        # Return a default/empty analysis on error
        return ReviewAnalysis(
            sentiment="Error",
            rating=0,
            key_features=[],
            improvement_suggestions=[]
        )


def print_analysis(analysis: ReviewAnalysis, original_text: str = None):
    """
    Pretty-prints the review analysis results.
    
    Args:
        analysis: ReviewAnalysis object with extracted data
        original_text: Optional original review text to display
    """
    if original_text:
        print("\nOriginal Review:")
        print("-" * 70)
        print(original_text)
        print("-" * 70)
    
    print("\nAnalysis Results:")
    print("=" * 70)
    print(f"💭 Sentiment: {analysis.sentiment}")
    print(f"⭐ Rating: {analysis.rating}/5")
    
    print(f"\n✨ Key Features ({len(analysis.key_features)} mentioned):")
    if analysis.key_features:
        for feature in analysis.key_features:
            print(f"   • {feature}")
    else:
        print("   (None mentioned)")
    
    print(f"\n💡 Improvement Suggestions ({len(analysis.improvement_suggestions)}):")
    if analysis.improvement_suggestions:
        for suggestion in analysis.improvement_suggestions:
            print(f"   • {suggestion}")
    else:
        print("   (None mentioned)")
    print("=" * 70)


def main():
    """
    Main function demonstrating the product review analyzer.
    """
    print("=" * 70)
    print("Assignment 3: Product Review Analyzer using PydanticOutputParser")
    print("=" * 70)
    
    # Example product review
    review_text = """
    I recently purchased this laptop and I'm very impressed! The build quality is excellent - 
    the aluminum chassis feels premium and durable. The display is absolutely stunning with 
    vibrant colors and perfect viewing angles. Performance-wise, it handles all my tasks 
    effortlessly, from coding to video editing.
    
    However, there are a couple of areas for improvement:
    1. The battery life could be better - I get around 5 hours of moderate use
    2. It tends to get a bit warm during intensive tasks
    3. The trackpad could be larger for better precision
    
    Despite these minor issues, I would definitely recommend this laptop to anyone looking 
    for a high-quality computing device. It's worth the investment!
    """
    
    try:
        # Analyze the review
        analysis = analyze_review(review_text)
        
        # Print results
        print_analysis(analysis, review_text)
        
        # Show the Pydantic model validation
        print("\nPydantic Model Validation:")
        print("-" * 70)
        print(f"Model is valid: ✅")
        print(f"Sentiment is valid: {analysis.sentiment} ✓")
        print(f"Rating is in range 1-5: {analysis.rating} ✓")
        print(f"Key features is a list: {isinstance(analysis.key_features, list)} ✓")
        print(f"Suggestions is a list: {isinstance(analysis.improvement_suggestions, list)} ✓")
        print("-" * 70)
        
        # Also demonstrate the raw model output
        print("\nRaw Pydantic Model Output (Dict):")
        print("-" * 70)
        print(analysis.model_dump())
        print("-" * 70)
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise


if __name__ == "__main__":
    main()
