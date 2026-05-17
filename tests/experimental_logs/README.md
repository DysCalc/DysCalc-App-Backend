# Experimental Logs: Diagnostic-to-Intervention Pipeline

## Overview
This directory contains the raw server logs, LM Studio logs, and validation outputs for the five-experiment study conducted on the DysCalc AI architecture. The goal of this study was to determine the minimum computational and parametric requirements necessary to reliably execute a Dual-Pass LLM pipeline (Pedagogical Drafting + JSON Schema Formatting) for Special Education (SPED) mathematics intervention generation.

## The Hypothesis vs. The Reality
The initial project proposal hypothesized that a 7-billion parameter math-specific model (`Qwen-Math-7B`) would be sufficient for both data privacy and pedagogical generation. 

Empirical testing proved this hypothesis incorrect. The experiments documented in this folder systematically isolated variables (Hardware VRAM vs. Model Intelligence) to discover the true architectural requirements for a safe, hallucination-free SPED intervention system.

## Experimental Progression 

| Exp | Architecture | Pass 1: Drafter | Pass 2: Formatter | Result | Primary Finding |
|:---|:---|:---|:---|:---|:---|
| **1** | Monolithic Local | `qwen2.5-math-7b` (LM Studio) | *None — single pass* |  **0% success** | Single-pass prompting causes cognitive overload. Model generates LaTeX, Chinese hallucinations, and invalid arithmetic simultaneously. |
| **2** | Two-Pass Local | `qwen2.5-math-7b` (LM Studio) | `qwen-2.5-7b-instruct` (OpenRouter*) |  **0% success** | Hardware context ceiling (4,096 tokens) makes repair loop destructive. Each retry grows the prompt, accelerating collapse. |
| **3** | Hybrid | `qwen2.5-math-7b` (LM Studio) | `qwen-2.5-72b-instruct` (OpenRouter) |  **0% success** | 72B formatter demonstrates robust recovery (9/10 items, 0 math errors in best attempt), but Pass 1 ceiling remains binding constraint. |
| **4** | Two-Pass Cloud Base | `qwen-2.5-7b-instruct` (OpenRouter) | `qwen-2.5-72b-instruct` (OpenRouter) |  **0% success** | Hardware constraint eliminated. New failure: 7B instruct has an intelligence ceiling — cannot maintain SPED domain ratio rules across attempts. |
| **5** | Two-Pass Cloud Ceiling | `qwen-2.5-72b-instruct` (OpenRouter) | `qwen-2.5-72b-instruct` (OpenRouter) |  **100% success** | First complete validation pass: 3/3 modules, 10/10 items, 0 math errors, 0 pedagogy errors, 0 schema errors. |

## Detailed Justifications for Architectural Shift

### 1. Why the Math-Specialized Model Failed
Initial testing prioritized `Qwen2.5-Math` because the domain is mathematics. However, as noted in the official Hugging Face model documentation, these models are specialized for solving equations and are not recommended for generalized instruction-following tasks. The pipeline requires Pass 1 to roleplay as a Senior SPED Teacher with domain-specific pedagogical rules — a task outside the Math model's fine-tune distribution. This caused the LaTeX bias and context degradation observed in Experiments 1 through 3.

### 2. Why Local Hardware Is Insufficient
The logs in `/exp1` through `/exp3` provide empirical evidence that a standard consumer GPU (RTX 3050, 4GB VRAM) cannot handle the context requirements of this system. With the unoptimized ML diagnostic payload reaching ~3,533–4,036 tokens, the model's 4,096-token context is exhausted before a complete response can be generated. The repair loop mechanism, which is designed to improve output quality, instead accelerates context collapse by growing the prompt on each retry.

### 3. The Intelligence Floor for SPED Safety
Experiment 4 proved that even with unlimited cloud memory and a correct general instruct model, a 7B parameter model is insufficient. The 7B model consistently failed to maintain the Addition vs. Subtraction Asymmetry domain ratio rule across all 5 attempts — a pedagogical safety requirement that prevents students from receiving subtraction-only practice (contraindicated by Geary, 2011, for dyscalculic learners). The 72B parameter tier is the empirically determined minimum for this application.

## Conclusion

This study constitutes a formal, empirically-grounded record of the architectural evolution from a local 7B proposal to a cloud 72B implementation. Each experiment's failure is not a research setback — it is a documented, reproducible data point that eliminates a specific failure variable and justifies the next architectural decision. The 100% success rate in Experiment 5 validates both the two-pass pipeline design and the 72B intelligence threshold as the correct solution for safe, hallucination-free SPED intervention generation.

**This documentation serves as a formal record for Chapter 4 (Results and Discussion) of the thesis, backed by complete raw logs, validation reports, and output JSONs for every experiment.**
