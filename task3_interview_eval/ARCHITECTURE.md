# Architecture: AI Interview Evaluation System

## System Overview

The AI Interview Evaluation System is a comprehensive, automated pipeline designed to evaluate a candidate's technical interview performance across three critical dimensions: Communication, Problem Solving, and Time Management. It aggregates various signals—ranging from NLP-derived confidence metrics to static code complexity and dynamic test results—into a unified score and actionable feedback report.

## Architecture Diagram (ASCII)

```text
  [Transcript]       [Code Solution]       [Time Data]
       |                    |                   |
       v                    v                   v
+--------------+    +--------------+    +--------------+
|SpeechAnalyzer|    | CodeAnalyzer |    |  TimeScorer  |
| - Fluency    |    | - Syntax     |    | (Evaluator)  |
| - Clarity    |    | - Complexity |    | - Ratio      |
| - Vocabulary |    | - Correctness|    | - Pacing     |
| - Confidence |    | - Style      |    |              |
| - Grammar    |    +--------------+    +--------------+
+--------------+            |                   |
       |                    |                   |
       +--------+           |           +-------+
                v           v           v
           +---------------------------------+
           |       InterviewEvaluator        |
           | (Weighted Sum: 40% / 40% / 20%) |
           +---------------------------------+
                            |
                            v
           +---------------------------------+
           |        FeedbackGenerator        |
           | (Strengths, Improvements, Plan) |
           +---------------------------------+
                            |
                            v
               [ Final Evaluation Report ]
```

## Component Descriptions

- **SpeechAnalyzer**: 
  - Parses and evaluates the transcript text.
  - **Models used**: Uses `spacy` for basic tokenization, `textstat` for reading metrics, `language_tool_python` for grammar. It also leverages Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english` for sentiment and `facebook/bart-large-mnli` for zero-shot tone classification) to accurately gauge confidence. 
  - *Tradeoff*: These models execute locally, requiring more initialization time but ensuring no data leaves the system.

- **CodeAnalyzer**: 
  - Analyzes the submitted code. 
  - **Why AST over regex?**: The `ast` module provides structural understanding (e.g., detecting nested loops or recursive calls), which is far more robust than regex for complexity analysis.
  - **Why subprocess sandbox?**: Code execution requires a sandbox. Using `subprocess.run` with a strict timeout prevents infinite loops and insulates the main evaluator from fatal crashes (e.g., SegFaults).
  
- **FeedbackGenerator**: 
  - **Rule-based vs Model-based**: Combines rule-based triggers (e.g., `if filler_count > 5`) for predictable, highly specific feedback with a zero-shot model (`bart-large-mnli`) to dynamically identify the *primary focus area*. This hybrid approach maximizes both reliability and personalization.

## Scoring Justification

| Category        | Weight | Rationale |
|-----------------|--------|-----------|
| Communication   | 40%    | In a real technical interview, articulating thought processes is often as important as the code itself. A brilliant but completely unexplainable solution often leads to rejection. |
| Problem Solving | 40%    | Correctness, efficiency, and clean code are the core technical requirements for the job. |
| Time Management | 20%    | Engineering requires delivering under constraints. This acts as a modifier—great performance within time is rewarded, while excessive overtime indicates struggling. |

## NLP Model Choices

- **DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)**: Chosen because it is extremely fast and retains 97% of BERT's performance while being 40% smaller. Perfect for extracting baseline sentiment without heavy GPU requirements.
- **BART MNLI (`facebook/bart-large-mnli`)**: Provides incredible zero-shot classification capabilities. Instead of fine-tuning a custom model for "confident" vs "hesitant" speech, we can classify tone on the fly.
- **Why NOT GPT API?**: 
  1. **Cost & Latency**: API calls cost money per interview and introduce unpredictable network latency.
  2. **No External Dependencies**: The prompt explicitly requires the system to run without external API keys or paid services.
  3. **Privacy**: Candidate data should remain local and secure.

## Data Flow & No External Dependencies

The system strictly utilizes local, pip-installable libraries. When a transcript or code snippet enters the system, it is processed entirely in memory or via secure local temporary files. 

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run tests:**
   ```bash
   python tests/test_all.py
   ```
3. **Run the End-to-End Demo:**
   ```bash
   python interview_evaluator.py
   ```

## Testing Strategy

- `tests/test_all.py` leverages Python's built-in `unittest` framework.
- It covers all major branches, including:
  - Sub-scorers (penalizing filler words, detecting O(n²) logic).
  - Robustness (graceful handling of empty strings or syntax errors).
  - Weighted arithmetic in the Evaluator.
  - The deterministic generation of study plans and actionable feedback.

## Limitations & Future Improvements

- **Fine-tuned Models**: Currently, the system uses zero-shot and pre-trained sentiment models. Fine-tuning a model specifically on technical interview transcripts would yield more precise confidence and tone detection.
- **Video/Audio Analysis**: Adding multi-modal analysis (facial expressions, eye contact, tone of voice via audio spectrograms) would drastically improve communication scoring.
- **Real-time Evaluation**: Moving from batch processing at the end of the interview to streaming evaluation would allow the system to act as an active, real-time interviewer.
