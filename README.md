# Assignment 3: Product Review Analyzer

## Overview

This assignment demonstrates the creation of an AI-powered product review analysis system using LangChain's PydanticOutputParser. The system analyzes customer reviews and extracts structured insights including sentiment, ratings, key features, and improvement suggestions into a validated Pydantic model.

## Features

- **Structured Data Extraction**: Uses Pydantic models for type-safe data extraction
- **Sentiment Analysis**: Identifies overall sentiment (Positive, Negative, Neutral)
- **Rating Extraction**: Extracts numerical ratings from 1-5 scale
- **Feature Identification**: Identifies key product features mentioned
- **Improvement Suggestions**: Extracts customer feedback for improvements
- **Data Validation**: Pydantic validators ensure data integrity and type safety
- **Error Handling**: Robust error handling with fallback responses

## Requirements

- Python 3.8+
- Ollama installed and running
- Mistral model pulled (`ollama pull mistral`)
- Required packages: langchain-core, langchain-ollama, pydantic

## Installation

1. Ensure Ollama is installed and running:
   ```bash
   ollama serve
   ```

2. Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

3. Install required Python packages:
   ```bash
   pip install langchain-core langchain-ollama pydantic
   ```

## Usage

Run the assignment:

```bash
python assignment3_review_analyzer.py
```

The program will:
1. Create a review analysis chain with Pydantic output parsing
2. Process a sample product review
3. Extract structured insights into a validated Pydantic model
4. Display formatted analysis results
5. Show model validation details

## Code Structure

### `ReviewAnalysis` Pydantic Model
- Defines the output schema with field types and validation
- Includes custom validators for sentiment and list fields
- Ensures data integrity through type checking and constraints

### `create_review_analyzer()`
- Initializes Ollama LLM with Mistral model
- Creates PydanticOutputParser with the ReviewAnalysis model
- Generates format instructions for structured output
- Returns a LangChain chain (prompt → LLM → parser)

### `analyze_review(review_text)`
- High-level function for review analysis
- Includes error handling and fallback responses
- Returns validated ReviewAnalysis object

### `print_analysis(analysis, original_text)`
- Pretty-prints analysis results with emojis and formatting
- Displays original review text alongside analysis

## Pydantic Model Schema

```python
class ReviewAnalysis(BaseModel):
    sentiment: str  # "Positive", "Negative", "Neutral", etc.
    rating: int     # 1-5 scale
    key_features: List[str]  # Product features mentioned
    improvement_suggestions: List[str]  # Areas for improvement
```

## Key Components

### PydanticOutputParser
- Ensures LLM outputs conform to Pydantic model schema
- Provides automatic type validation and conversion
- Generates format instructions for the LLM
- Handles complex nested data structures

### Custom Validators
- **Sentiment Validator**: Accepts valid sentiment values with flexibility
- **List Validator**: Ensures fields are always lists, converting strings if needed
- **Field Constraints**: Rating limited to 1-5 range, sentiment length constraints

## Learning Objectives

- Understanding PydanticOutputParser for structured data extraction
- Creating and validating Pydantic models with custom validators
- Implementing type-safe data structures for LLM outputs
- Error handling in structured parsing pipelines
- Building robust review analysis systems

## Example Output

For a laptop review, the system extracts:
- **Sentiment**: "Positive" (with some negative aspects)
- **Rating**: 4/5
- **Key Features**: ["build quality", "display", "performance"]
- **Improvements**: ["battery life", "heat management", "trackpad size"]

The Pydantic model ensures all data is properly typed and validated, preventing runtime errors and ensuring data consistency.

## Advanced Features

- **Type Safety**: All extracted data is validated against the schema
- **Fallback Handling**: Graceful error recovery with default values
- **Extensible Schema**: Easy to add new fields or validation rules
- **JSON Serialization**: Automatic conversion to/from JSON format
- **Model Validation**: Runtime checking of data constraints</content>
<parameter name="filePath">c:\Users\jhanv\OneDrive\Documents\ollama\README_assignment3.md
