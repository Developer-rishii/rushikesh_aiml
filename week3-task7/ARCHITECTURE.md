# System Architecture

```mermaid
graph TD
    A[Data Generator] -->|Outputs Candidates & Jobs| B(Data Directory)
    A -->|Outputs Pairs| B
    
    B -->|Reads Data| C[Features Extractor]
    B -->|Reads Data| D[Baseline Logic]
    
    C -->|Engineered Features| E[Training Script]
    E -->|Trains| F(Logistic Regression Model)
    E -->|Logs| G(Experiment Tracking JSONL)
    
    C -->|Feature Generation| H[Core Matcher]
    D -->|Skill Overlap| H
    F -->|Model Probability| H
    
    H -->|Combines Scores & Explains| I[Evaluation Script]
    H -->|Combines Scores & Explains| J[Walkthrough Demo]
    H -->|Combines Scores & Explains| K[FastAPI Service]
    
    B -->|Reads Data| I
    B -->|Reads Data| J
    B -->|Reads Jobs| K
    
    User[Client / User] -->|POST /match| K
    K -->|Ranked Jobs & Explanations| User
```

## Flow Description
1. **Data Generation**: `data_generator.py` generates the synthetic mock tables for training and testing.
2. **Feature Engineering**: `features.py` centralizes feature extraction logically matching skills, education, experience, and project counts.
3. **Training**: `train.py` takes engineered features, trains Logistic Regression and produces the `.joblib` model object.
4. **Matcher Core**: `matcher.py` combines the rules-based baseline score with ML probabilities and constructs an explainable payload.
5. **Services & Endpoints**: Both the offline testing scripts and the live API endpoint consume the Matcher Core logic.
