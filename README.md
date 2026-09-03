<div align="center">

<img width="1799" height="504" alt="image" src="https://github.com/user-attachments/assets/a6ced70e-e695-484d-8809-7580141a536a" />

# LeaseGuard
LeaseGuard is an open-source, domain-adapted 8B LLM fine-tuned for commercial real estate lease extraction, multi-jurisdictional risk scoring, and automated clause redlining. Engineered for precision structured JSON output and low-latency micro-inference—delivering contract analysis at a fraction of frontier model API costs

</div>

## Overview

LeaseGuard is a small 8B parameter language model. It analyzes commercial real estate leases. Standard models can make errors with complex legal documents. LeaseGuard gives accurate contract analysis without expensive API costs.

- **Base Models:** Llama-3.1-8B-Instruct or Qwen-2.5-7B-Instruct
- **Target Users:** Property management companies, real estate operations, and law firms
- **Deployment:** On-premise hardware or low-cost cloud GPU instances

## Key Functions

- **Data Extraction:** Extracts complex lease clauses into structured JSON format.
- **Risk Assessment:** Finds hidden liabilities, indemnification limits, and termination penalties.
- **Clause Redlining:** Recommends alternative contract text for negotiation.
- **Cost Reduction:** Operates at 1/15th the token cost of frontier LLM APIs.

## Input and Output Example

### Input Prompt

Extract tenant renewal options and early termination penalties from Section 14.2:

### Model Output

```json
{
  "renewal_clause": {
    "option_period_years": 5,
    "notice_deadline_days": 180,
    "rate_structure": "Fair Market Value with 3% annual floor"
  },
  "termination_penalty": {
    "allowed": true,
    "penalty_fee": "6 months unamortized tenant improvements",
    "risk_level": "MEDIUM"
  }
}

```

## Model Training

LeaseGuard uses QLoRA fine-tuning on Apple Silicon hardware (MLX) or cloud GPUs (Unsloth).

* **Training Size:** 1,200 curated commercial lease examples
* **Quantization:** 4-bit loading ($r=16$, $\alpha=32$)
* **Data Source:** SEC EDGAR 10-K commercial lease filings

## License

This project uses the MIT License.
