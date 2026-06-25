# PersAgent Multi-Agent Runtime

PersAgent now keeps the original knowledge base and canonical tools under
`tool/`, while using a PersAgent-native LangChain/LangGraph-style agent runtime
for two-stage personalized toxicity reasoning.

## Runtime layout

```text
agents/
  brain_agent.py              # input parsing, two-stage reasoning, report synthesis
  input_parser.py              # input normalization through tool.common.resolve_drug
  knowledge_retrieval_agent.py # tool planning, tool calls, evidence packaging
  verifier_agent.py           # deterministic schema/content/safety checks
  mechanism_chain_builder.py   # drug -> metabolism -> target -> pathway -> ADE chains
graph/
  graph_builder.py            # LangGraph StateGraph workflow
  nodes.py                    # graph node functions
models/
  formatting.py                # report JSON formatting/runtime serialization
  schemas.py                  # Pydantic report/evidence schemas
  state.py                    # shared graph state
tool/
  agent_runtime.py             # PersAgent-native runtime adapter over current tools
main.py                       # warfarin demo entry point
```

## Conflict rule

PersAgent remains the source of truth for data and tool behavior. The runtime
adapter in `tool/agent_runtime.py` calls existing PersAgent tools such as:

- `tool.mechanism_admet.admetsar_predict`
- `tool.mechanism_admet.dti_query`
- `tool.mechanism_admet.mechanism_query`
- `tool.mechanism_admet.pathway_enrich`
- `tool.mechanism_admet.drugbank_metabolism_query`
- `tool.mechanism_admet.ddi_query`
- `tool.ade_profile.persade_drug_profile`
- `tool.ade_profile.persade_contextual_retrieval`

The old PerTox-agent tool layer is not used by the runtime. Agent-only helpers
live under `agents/` and `models/`; knowledge and retrieval behavior lives under
the current PersAgent `tool/` package.

## Run

```bash
python main.py
```

The demo writes:

```text
outputs/final_report_warfarin.json
```

Run the project from an environment with `langgraph` installed, such as the
`perstox` Conda environment.

## Organ scope

Stage 1 and Stage 2 keep all eight SOC rows in the output schema, but only
liver and heart are actively modeled. Kidney, hematologic, immune, skin,
neurologic, and gastrointestinal rows are retained as null placeholders.

## Attribution mode

When live LLM mode is enabled, molecular attribution for liver and heart is
generated directly from the retrieved EvidencePackage context: compressed
`tool_results`, organ-relevant `evidence_items`, all retrieved evidence item
summaries, and the baseline probability audit. Local candidate-driver rules are
used only as a deterministic fallback when no live LLM result is available.

## Live LLM

The runtime defaults to DeepSeek-compatible live LLM configuration, while
keeping deterministic fallback behavior when `PERSAGENT_USE_LIVE_LLM` is not
enabled or no API key is present.

Create a local `.env` from `.env.example` and fill in your key:

```text
PERSAGENT_USE_LIVE_LLM=true
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=replace_with_your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_ATTRIBUTION_PARALLELISM=2
```

To switch later to GPT/OpenAI:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=replace_with_your_openai_api_key
LLM_MODEL=gpt-4o
```
