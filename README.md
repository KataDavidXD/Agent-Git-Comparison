# Agent Framework Comparison for Literature Review Generation

> **Related Project**: This experimental framework was designed to evaluate the AgentGit framework for academic research. AgentGit project: [[Agent-Git](https://github.com/KataDavidXD/Agent-Git.git)]

## Experimental Overview

This repository implements a comprehensive empirical study comparing four different AI agent frameworks for automated literature review generation. The experiment evaluates how different agent architectures affect the quality of generated academic summaries using standardized evaluation metrics.

## Experimental Purpose

The primary objectives of this research are:

1. **Framework Performance Comparison**: Systematically evaluate the effectiveness of different multi-agent frameworks (AgentGit, LangGraph, AutoGen, Agno) for literature review tasks
2. **Prompt Strategy Analysis**: Investigate how different prompting strategies (Chain-of-Thought vs Few-Shot learning) impact output quality across frameworks
3. **Quality Assessment**: Develop robust evaluation metrics using G-Eval methodology to assess fluency, coherence, consistency, and relevance
4. **Reproducibility Study**: Conduct large-scale experiments (20 runs per framework) to ensure statistical significance and reliability

## Experimental Methodology

### 1. Agent Frameworks Under Study

#### AgentGit
- **Architecture**: Uses AgentGit's checkpoint branching system
- **Key Features**: Explicit branch management with rollback capabilities
- **Workflow**: Search → Extract → Introduction → Analysis → Discussion → Concatenation
- **Branching Strategy**: Creates multiple branches testing different prompt combinations

#### LangGraph Time Travel
- **Architecture**: LangGraph's native checkpointing and time travel functionality
- **Key Features**: State graph execution with memory-based checkpointing
- **Workflow**: Same 5-phase structure but uses LangGraph's time travel for prompt testing
- **Branching Strategy**: Rewinds to checkpoints and continues with different prompts

#### AutoGen Multi-Agent
- **Architecture**: Microsoft AutoGen framework with specialized agents
- **Key Features**: Agent-to-agent communication and role-based task distribution
- **Workflow**: Asynchronous execution with dedicated search, analysis, and synthesis agents
- **Branching Strategy**: Different agent configurations for prompt variation testing

#### Agno Workflow Engine
- **Architecture**: Agno's step-based workflow system
- **Key Features**: Structured workflow execution with SQLite state management
- **Workflow**: Step-by-step processing with configurable prompt injection points
- **Branching Strategy**: Workflow variations for different prompt combinations

### 2. Experimental Design

#### Phase 1: Data Collection
- **Source**: arXiv API for academic papers
- **Topic**: "Large Language Models" (configurable)
- **Papers**: 50 recent papers per experiment
- **Data Processing**: Abstract extraction and structured formatting

#### Phase 2: Prompt Strategy Testing
Each framework tests two distinct prompting approaches:

**Chain-of-Thought (CoT)**:
- Explicit step-by-step reasoning
- "Think before you write" methodology
- Reasoning transparency requirements

**Few-Shot Learning**:
- Example-based guidance
- Pattern matching from provided templates
- Structural consistency emphasis

#### Phase 3: Multi-Branch Execution
- **Branch Generation**: Each framework creates multiple execution branches
- **Prompt Combinations**: Systematic testing of Introduction×Analysis×Discussion prompts
- **Parallel Processing**: Concurrent execution where supported by framework

### 3. Evaluation Framework

#### G-Eval Implementation
Based on the G-Eval paper methodology:
- **Model**: GPT-4o-mini for evaluation
- **Sampling**: 20 samples per evaluation (n=20)
- **Temperature**: 2.0 for robust scoring
- **Dimensions**: Four-dimensional assessment

#### Evaluation Metrics

**Fluency**: 
- Grammatical correctness
- Natural language flow
- Readability assessment

**Coherence**:
- Logical structure
- Argument progression
- Section connectivity

**Consistency**:
- Factual accuracy
- Citation reliability
- Internal logical consistency

**Relevance**:
- Topic alignment
- Research question focus
- Scope appropriateness

#### Statistical Analysis
- **Runs per Framework**: 20 independent executions
- **Aggregation**: Mean, standard deviation, min/max statistics
- **Comparison**: Cross-framework performance analysis
- **Visualization**: Comprehensive charts and statistical summaries

### 4. Experimental Pipeline

```
1. Topic Input → 2. Paper Search → 3. Abstract Extraction
                        ↓
4. Framework Execution → 5. Branch Generation → 6. Content Synthesis
                        ↓
7. G-Eval Assessment → 8. Statistical Analysis → 9. Comparative Visualization
```

## Technical Implementation

### Core Components

- **Experiment Runner** (`run_experiments.py`): Orchestrates all framework comparisons
- **Framework Agents**: Individual implementations for each agent framework
- **Evaluator System**: G-Eval based assessment with multi-dimensional scoring
- **Statistical Analysis**: Automated comparison and visualization generation
- **Result Aggregation**: Cross-run statistics and framework comparison

### Key Features

- **Parallel Execution**: Multi-threaded experiment running for efficiency
- **Retry Logic**: Robust error handling with exponential backoff
- **Token Tracking**: Comprehensive usage monitoring across all frameworks
- **Reproducibility**: Seeded random processes and detailed logging
- **Extensibility**: Modular design for adding new frameworks or evaluation metrics

## Results and Analysis

The experiment generates comprehensive evaluation reports including:

1. **Framework Performance Rankings**: Comparative analysis across all quality dimensions
2. **Prompt Strategy Effectiveness**: CoT vs Few-Shot performance comparison
3. **Statistical Significance**: Confidence intervals and hypothesis testing
4. **Resource Utilization**: Token usage and execution time analysis
5. **Quality Distributions**: Score variance and consistency metrics

## Usage

Run the complete experimental suite:

```bash
python test/run_experiments.py
```

This will:
- Execute 20 runs for each of the 4 frameworks
- Generate evaluation summaries for each framework
- Create comparative analysis reports
- Produce visualization outputs

## Dependencies

- **Agent Frameworks**: agent_git, autogen, langgraph, agno
- **Evaluation**: OpenAI API for G-Eval implementation
- **Analysis**: pandas, numpy, matplotlib, seaborn
- **Utilities**: requests, json, pathlib, concurrent.futures

## Research Contributions

This experimental framework contributes to understanding:
- The relative strengths of different multi-agent architectures
- The impact of prompting strategies on literature review quality
- Scalable evaluation methodologies for academic text generation
- Reproducible benchmarking approaches for AI agent systems