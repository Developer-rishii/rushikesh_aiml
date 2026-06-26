# Spend-Quality Guardrail Architecture

This document describes the flow from the existing matching model to the new spend-quality guardrail.

```mermaid
flowchart TD
    subgraph Upstream [Task 6: Matching Pipeline]
        CP[Candidate Profile]
        JP[Job Posting]
        BM{Baseline Matcher}
        CP --> BM
        JP --> BM
        BM -->|Match Score + Data| GC
    end

    subgraph Guardrail [Task 8: Spend-Quality Guardrail]
        GC[Guardrail Check API]
        TC[(Calibrated Threshold)]
        EXP[Explainability Engine]
        
        GC -->|Compare Score| TC
        TC -->|Pass/Fail| EXP
        EXP -->|Generate Reasons| RES[Guardrail Decision\n(OK / LOW_FIT_WARNING)]
    end
    
    subgraph Downstream [Spend Protection Flow]
        RES -->|If OK| APP[Authorize Application Spend]
        RES -->|If LOW_FIT_WARNING| BLOCK[Block Payment & Show Warning]
    end
```

## Handoff Boundaries
- **Matching Handoff**: The guardrail assumes the matching score and basic feature overlaps (like matched/missing skills) are already computed by the upstream matcher.
- **Spend Protection Handoff**: This guardrail does not interact with Stripe/payments. It simply outputs a JSON decision that the frontend or payment gateway must check before initiating a charge.
