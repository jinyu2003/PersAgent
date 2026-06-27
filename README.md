# PerTox-agent

**面向药物毒性预测的两阶段 Agent 知识库与工具层。**

PerTox-agent 是一个基于 LangGraph 的多 Agent 药物毒性推理框架，将分散的药理/毒理/药物基因组学数据源组织成 Agent 可检索的本地知识库，
并在其上实现一组工具，支撑两阶段推理：

- **Stage 1：通用毒性推理**
  从药物结构、ADMET 端点、药物-靶点相互作用、代谢/转运、机制通路和已知 ADE 信号中估计人群层面的器官/SOC 毒性基线。

- **Stage 2：个性化毒性推理**
  在 Stage 1 基线基础上，叠加患者年龄、性别、肝肾功能、基因型、HLA、合并疾病和合并用药等因素，生成个体化风险偏移、机制修饰和临床监测建议。

设计强调**本地优先、可复现、可审计**：原始数据只读，派生表带 provenance（输入哈希、行数、
schema），API 响应 cache-first 落盘，跨库统一以 **InChIKey** 为连接键。






## 系统架构

```text
raw patient / drug input
        |
        v
Input Normalization
        |
        v
Toxicity Orchestrator Agent
        |
        +--> Stage 1 Retrieval Plan
        |           |
        |           v
        |   Knowledge Retrieval Agent
        |           |
        |           v
        |   EvidencePackage
        |           |
        |           v
        |   UniversalToxicityReport
        |
        +--> Patient Profile Standardization
        |
        +--> Stage 2 Retrieval Plan
                    |
                    v
            Knowledge Retrieval Agent
                    |
                    v
            PersonalizedToxicityReport
                    |
                    v
            Safety Verifier Agent
                    |
                    v
              Final JSON Report
```

## Agent 角色

### 1. Toxicity Orchestrator Agent

负责输入解析、阶段规划、毒性推理和报告合成。

主要职责：

- 解析原始患者信息和药物信息。
- 规划 Stage 1 通用毒性所需证据。
- 规划 Stage 2 个体化修饰所需证据。
- 合并证据包并生成通用毒性报告。
- 基于患者因素生成个体化毒性报告。
- 将最终结果格式化为结构化 JSON。

对应代码：

```text
src/pertox_agent/agents/toxicity_orchestrator_agent.py
```

### 2. Knowledge Retrieval Agent

负责工具规划、工具调用、证据封装和冲突标记。

主要职责：

- 将 Orchestrator 的结构化查询转换为确定性工具调用。
- 调用本地知识库和可选 API 工具。
- 生成统一的 EvidencePackage。
- 标记药物标准化失败、工具失败、证据冲突等问题。
- 不直接做临床决策。

对应代码：

```text
src/pertox_agent/agents/knowledge_retrieval_agent.py
```

### 3. Safety Verifier Agent

负责独立安全校验，不产生新的毒性推理。

主要职责：

- 检查输出 schema 是否完整。
- 检查概率、CTCAE grade、SOC 行等格式约束。
- 检查重要证据是否被建议文本反映。
- 检查确定性安全红线，如禁忌 DDI、妊娠期 warfarin、abacavir 与 HLA-B*57:01 等。
- 对高风险低置信度结果进行标记。

对应代码：

```text
src/pertox_agent/agents/safety_verifier_agent.py
```

## 知识库与工具层

PerTox-agent 使用**本地优先、API 补充、运行时统一适配**的知识策略。底层工具按职责放在
`src/pertox_agent/tools/` 的不同子包中；Agent 不直接依赖这些文件名，而是通过
`src/pertox_agent/tools/runtime/retrieval_runtime.py` 暴露的一组稳定检索函数进行调用。

当前工具层分为 7 个部分：

| 分层 | 路径 | 作用 |
|---|---|---|
| 临床输入解析 | `src/pertox_agent/tools/clinical_input/` | 解析原始患者/药物输入，融合确定性解析与可选 LLM JSON 抽取 |
| 患者上下文 | `src/pertox_agent/tools/patient_context/` | 标准化患者画像，解析适应症、PGx phenotype、器官功能分层 |
| 分子证据 | `src/pertox_agent/tools/molecular_evidence/` | ADMET、DTI、DDI、代谢/转运、机制证据、通路富集 |
| 真实世界证据 | `src/pertox_agent/tools/real_world_evidence/` | PersADE / FAERS-derived ADE 全谱、相似病例、亚组风险 |
| 毒性归因 | `src/pertox_agent/tools/toxicity_attribution/` | 构建 drug -> metabolism -> target -> pathway -> ADE 机制链 |
| 共享数据访问 | `src/pertox_agent/tools/shared/` | 药物解析、InChIKey 桥接、DrugBank client、缓存与通用读写逻辑 |
| 运行时适配 | `src/pertox_agent/tools/runtime/` | 面向 Agent 的统一检索接口，将底层工具结果整理为报告友好的 payload |

底层工具通常保留统一调用形式：

```python
run(payload) -> dict
```

但在多 Agent pipeline 中，`KnowledgeRetrievalAgent` 调用的是运行时函数：

| 运行时函数 | 主要功能 | 对应底层模块 |
|---|---|---|
| `drug_card_lookup` | 药物标准化、结构、DrugBank-like 药物卡片 | `src/pertox_agent/tools/shared/` |
| `admetsar_predict` | ADMET 与毒性端点预测 | `src/pertox_agent/tools/molecular_evidence/admet_predictor.py` |
| `dti_query` | 药物-靶点相互作用 | `src/pertox_agent/tools/molecular_evidence/drug_target_interaction.py` |
| `pathway_enrich` | 靶点/基因集合通路富集 | `src/pertox_agent/tools/molecular_evidence/pathway_enrichment.py` |
| `mechanism_query` | ADE / 靶点 / 通路机制证据 | `src/pertox_agent/tools/molecular_evidence/mechanism_evidence.py` |
| `mechanism_chains_lookup` | Drug-Target-Pathway-ADE 归因链 | `src/pertox_agent/tools/toxicity_attribution/toxicity_chain_builder.py` |
| `persade_drug_profile` | 药物 ADE 全谱、人群信号、baseline organ score | `src/pertox_agent/tools/real_world_evidence/persade_drug_ade_profile.py` |
| `drugbank_metabolism_query` | 代谢酶、转运体、消除路径 | `src/pertox_agent/tools/molecular_evidence/drug_metabolism.py` |
| `ddi_query` | 合并用药相互作用筛查 | `src/pertox_agent/tools/molecular_evidence/drug_drug_interaction.py` |
| `cpic_lookup` | PGx/CPIC 风险规则 | `src/pertox_agent/tools/runtime/retrieval_runtime.py` 本地规则 |
| `hla_peptide_score` | HLA 风险接口/占位规则 | `src/pertox_agent/tools/runtime/retrieval_runtime.py` 本地规则 |
| `persade_contextual_retrieval` | 相似患者上下文 ADE 检索 | `src/pertox_agent/tools/real_world_evidence/persade_similar_case_retrieval.py` |
| `persade_subgroup_scores` | 亚组 ADE 风险偏移 | `src/pertox_agent/tools/real_world_evidence/persade_subgroup_risk.py` |
| `similar_case_retrieval` | 相似病例包装接口 | runtime wrapper |
| `cohort_outcomes_query` | 队列结局包装接口 | runtime wrapper |

Stage 1 默认检索药物卡片、代谢、ADMET、DTI、机制、通路、PersADE 药物 ADE 谱和机制链；
Stage 2 默认检索 PGx、DDI、HLA、相似患者、亚组风险、相似病例和队列结局证据。

详细工具说明见：

```text
src/pertox_agent/tools/README.md
```

## 目录结构

```text
PerTox-agent/
  configs/
    llm.json
    kb_sources.json

  data/
    raw/
    normalized/
    cache/
    manifests/
    PersADE/

  docs/
  examples/
    warfarin.md
    run_warfarin_demo.py
  results/
  scripts/

  src/
    pertox_agent/
      __init__.py
      graph.py
      nodes.py
      state.py
      schemas.py
      formatting.py
      settings.py

      agents/
        toxicity_orchestrator_agent.py
        knowledge_retrieval_agent.py
        safety_verifier_agent.py

      tools/
        clinical_input/
        patient_context/
        molecular_evidence/
        real_world_evidence/
        toxicity_attribution/
        shared/
        runtime/

      kb_builder/
        downloader.py
        normalize.py
        api_cache.py
        manifest.py

  tests/
  langgraph.json
  pyproject.toml
  requirements.txt
  .env.example
```

## 环境安装

推荐使用 Python 3.10 或更高版本。

RDKit 建议通过 conda-forge 安装：

```bash
conda create -n perstox python=3.10 -c conda-forge rdkit
conda activate perstox
pip install -r requirements.txt
```

主要依赖包括：

```text
rdkit
pydantic
langgraph
langchain
langchain-openai
openai
typing-extensions
```

## 数据准备

本仓库默认采用代码与数据分离策略。数据文件不建议直接纳入 git。

如已获得项目数据包，可在项目根目录解压：

```bash
tar --use-compress-program=unzstd -xf perstox-data-*.tar.zst
```

解压后应包含类似目录：

```text
data/normalized/
data/raw/
data/cache/
```

部分数据源需要用户自行准备：

| 数据源 | 推荐位置 | 用途 |
|---|---|---|
| PersADE | `data/PersADE/` | ADE 信号、人群画像、机制链 |
| admetSAR 3.0 | `data/admetsar3_all_endpoints.txt` | ADMET 与毒性端点 |
| DrugBank 原始 XML | `data/raw/drugbank/` | 药物身份、靶点、代谢、DDI |
| MedDRA | 用户授权路径 | ADE/SOC 标准化 |
| CPIC / DPWG | `data/normalized/` | PGx 规则 |

如果需要从 DrugBank XML 重新构建派生表，可运行：

```bash
python -m pertox_agent.tools.shared.drugbank_client build
```

## 快速开始

运行内置 warfarin 示例：

```bash
python examples/run_warfarin_demo.py
```

示例输入包括：

- 药物：warfarin
- 患者：65 岁女性
- 肝功能：Child-Pugh B，ALT/AST 升高
- 肾功能：eGFR 45 mL/min
- 基因型：CYP2C9 *2/*3
- 合并用药：amiodarone
- 合并疾病：cirrhosis，atrial fibrillation

输出文件：

```text
results/final_report_warfarin.json
```

## Live LLM 模式

系统支持确定性本地 fallback，也支持启用 live LLM 生成机制归因叙述。

非敏感模型参数放在 `configs/llm.json`，例如 provider、模型名、base URL、max tokens、
temperature 和 attribution 并发数：

```json
{
  "provider": "deepseek",
  "use_live_llm": true,
  "models": {
    "default": "deepseek-v4-flash",
    "brain_model": "deepseek-v4-flash",
    "knowledge_model": "deepseek-v4-flash",
    "verifier_model": "deepseek-v4-flash"
  },
  "generation": {
    "max_tokens": 2048,
    "temperature": 0,
    "attribution_parallelism": 2
  }
}
```

密钥和机器本地覆盖放在 `.env`。复制环境变量模板：

```bash
cp .env.example .env
```

DeepSeek 密钥示例：

```text
DEEPSEEK_API_KEY=replace_with_your_key
```

如果要临时覆盖 `configs/llm.json`，仍可在 `.env` 中设置：

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=replace_with_your_key
LLM_MODEL=gpt-4o
```

配置优先级为：`.env` / 环境变量 > `configs/llm.json` > `src/pertox_agent/settings.py` 内置默认值。
如果未启用 live LLM 或未提供 API key，系统会使用确定性 fallback 逻辑继续运行。

## 输出格式

最终 JSON 报告包含以下核心部分：

```text
drug_entity
patient_features
structure_profile
universal_toxicity_report
personalized_toxicity_report
evidence_package
verification_report
```

其中 Stage 1 输出为：

```text
UniversalToxicityReport
```

Stage 2 输出为：

```text
PersonalizedToxicityReport
```

每个毒性条目尽量包含：

- SOC / 器官系统
- baseline risk level
- baseline probability
- personalized probability
- risk shift
- CTCAE grade
- molecular attribution
- patient attribution
- evidence references
- clinical recommendation
- uncertainty / limitation

当前版本保留 8 个 SOC/器官系统行：

```text
liver
heart
kidney
hematologic
immune
skin
neurologic
gastrointestinal
```

当前主动建模范围主要为 liver 与 heart，其余系统作为 placeholder 保留，便于后续扩展和 schema 稳定。

## 运行测试

运行工具层 smoke test：

```bash
python tests/test_tools.py
```

运行阶段性测试：

```bash
python tests/test_stage1_pipeline.py
python tests/test_stage2_context.py
python tests/test_agent_runtime.py
```

测试主要覆盖：

- 药物输入标准化
- 工具运行与返回格式
- Stage 1 证据检索与报告生成
- Stage 2 患者画像标准化
- Agent runtime 适配层
- 安全校验逻辑

## 证据等级与可审计性

系统通过 EvidencePackage 聚合工具返回结果。每条 evidence item 通常包含：

```text
tool_name
evidence_level
finding
strength
payload
citations
```

证据等级示例：

| 等级 | 含义 |
|---|---|
| `DrugCard` | 药物卡片或本地标准化药物知识 |
| `ADMET` | 结构/ADMET 模型或端点预测 |
| `P1` | 人群 ADE 信号或高优先级本地证据 |
| `P2` | 患者上下文、亚组或相似人群证据 |
| `P3` | PGx / CPIC 等规则证据 |
| `P4` | DDI 或临床规则证据 |
| `P5` | fallback、缺失、失败或低置信度证据 |

## 设计原则

### 1. 工具结果优先于自由生成

LLM 不直接访问外部知识，也不凭空生成证据。所有外部事实应来自 Knowledge Retrieval Agent 封装后的 EvidencePackage。

### 2. 本地知识优先

DrugBank、PersADE、admetSAR、Reactome、GO、DDInter2 等知识源优先使用本地快照，以提高复现性和降低 API 不稳定性。

### 3. API 只作为补充

联网 API 采用 cache-first 策略。API 失败时，工具应返回可恢复错误或降级到本地结果，而不是中断整个 pipeline。

### 4. 安全校验独立

Safety Verifier Agent 不生成新推理，只检查结构完整性、证据一致性和确定性红线规则。

### 5. 输出 schema 稳定

即使某些器官系统尚未主动建模，也保留对应 SOC 行，方便比较、评估和后续扩展。

## 局限性

当前版本仍有以下限制：

- 主动建模器官主要为 liver 与 heart。
- PersADE / FAERS 类信号代表统计关联，不等价于因果证明。
- 个体化风险计算依赖规则和可用患者信息，不能替代临床风险评估。
- LLM 归因文本仅作为证据包驱动的解释生成，不是新的独立证据来源。
- 数据源受各自授权限制，复现实验需要用户具备相应数据访问权限。
- 部分工具为占位或轻量实现，后续可接入更完整的临床规则、药物基因组学表和真实世界队列。

## 负责任使用声明

PerTox-agent 仅用于科研、教学和方法学验证。

请勿将本系统用于：

- 自动诊断
- 自动处方
- 替代医生或药师决策
- 对真实患者进行未经审查的临床建议
- 绕过药品说明书、指南或伦理审查

任何涉及真实患者的使用都应经过专业人员审核，并遵循当地法律法规、伦理规范和数据隐私要求。

## 数据许可

本仓库代码按项目 LICENSE 使用。

数据源遵循各自原始许可。使用者需自行确认合规性，尤其包括：

- DrugBank：通常需要学术或商业授权，原始数据不可随意再分发。
- MedDRA：需要 MSSO 授权。
- WHO ATC/DDD：需遵守 WHOCC 使用条款。
- PersADE：遵循提供方或自有数据授权。
- admetSAR：遵循其官方数据与服务条款。
- Reactome / GO / UniProt / HGNC / NCBI Gene / DDInter2 / CPIC / DPWG：使用时需保留来源说明和许可要求。



## 相关文档

```text
src/pertox_agent/tools/README.md
configs/llm.json
configs/kb_sources.json
docs/
```

## 致谢

本项目使用或参考了多个开放科学与药物安全数据资源。感谢相关数据库、工具和社区对药物安全、毒理学和可复现 AI 研究的支持。


