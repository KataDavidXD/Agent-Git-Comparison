"""Prompt templates for G-Eval evaluation dimensions.

Based on prompts from the G-Eval paper and SummEval dataset.
"""

from config.models import EvaluationDimension


# Summarization evaluation dimensions (from G-Eval/SummEval)

FLUENCY = EvaluationDimension(
    name="fluency",
    description="Quality of grammar, spelling, punctuation, word choice, and sentence structure",
    min_score=1,
    max_score=3,
    criteria="""Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.""",
    evaluation_steps=[],
    prompt_template="""You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.


Example:

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Fluency (1-3):"""
)


COHERENCE = EvaluationDimension(
    name="coherence",
    description="Collective quality of all sentences - well-structured and organized",
    min_score=1,
    max_score=5,
    criteria="""Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.\"""",
    evaluation_steps=[
        "Read the news article carefully and identify the main topic and key points.",
        "Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.",
        "Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria."
    ],
    prompt_template="""You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Example:


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Coherence:"""
)


CONSISTENCY = EvaluationDimension(
    name="consistency",
    description="Factual alignment between the summary and source - no hallucinations",
    min_score=1,
    max_score=5,
    criteria="""Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.""",
    evaluation_steps=[
        "Read the news article carefully and identify the main facts and details it presents.",
        "Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.",
        "Assign a score for consistency based on the Evaluation Criteria."
    ],
    prompt_template="""You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. 

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.


Example:


Source Text: 

{{Document}}

Summary: 

{{Summary}}


Evaluation Form (scores ONLY):

- Consistency:"""
)


RELEVANCE = EvaluationDimension(
    name="relevance",
    description="Selection of important content from source - no redundancies or excess information",
    min_score=1,
    max_score=5,
    criteria="""Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.""",
    evaluation_steps=[
        "Read the summary and the source document carefully.",
        "Compare the summary to the source document and identify the main points of the article.",
        "Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.",
        "Assign a relevance score from 1 to 5."
    ],
    prompt_template="""You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Example:


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Relevance:"""
)


# Agent/QA evaluation dimensions

HELPFULNESS = EvaluationDimension(
    name="helpfulness",
    description="How helpful the answer is in addressing the user's question",
    min_score=1,
    max_score=5,
    criteria="""Helpfulness (1-5) - how well the answer addresses the user's question and provides useful information.

- 1: Not helpful. The answer does not address the question or provides incorrect information.
- 2: Slightly helpful. The answer partially addresses the question but misses key points.
- 3: Moderately helpful. The answer addresses the question but could be more complete.
- 4: Very helpful. The answer thoroughly addresses the question with minor gaps.
- 5: Extremely helpful. The answer comprehensively addresses the question with accurate and useful information.""",
    evaluation_steps=[
        "Read the question carefully and understand what is being asked.",
        "Read the answer and evaluate how well it addresses the question.",
        "Assess the completeness, accuracy, and usefulness of the information provided.",
        "Assign a helpfulness score from 1 to 5."
    ],
    prompt_template="""You will be given a question and an answer to that question.

Your task is to rate the answer on one metric.

Please make sure you read and understand these instructions carefully.

Evaluation Criteria:

Helpfulness (1-5) - how well the answer addresses the user's question and provides useful information.

- 1: Not helpful. The answer does not address the question or provides incorrect information.
- 2: Slightly helpful. The answer partially addresses the question but misses key points.
- 3: Moderately helpful. The answer addresses the question but could be more complete.
- 4: Very helpful. The answer thoroughly addresses the question with minor gaps.
- 5: Extremely helpful. The answer comprehensively addresses the question with accurate and useful information.

Evaluation Steps:

1. Read the question carefully and understand what is being asked.
2. Read the answer and evaluate how well it addresses the question.
3. Assess the completeness, accuracy, and usefulness of the information provided.
4. Assign a helpfulness score from 1 to 5.


Example:

Question:

{{Question}}

Answer:

{{Answer}}


Evaluation Form (scores ONLY):

- Helpfulness (1-5):"""
)


CORRECTNESS = EvaluationDimension(
    name="correctness",
    description="Factual accuracy of the answer",
    min_score=1,
    max_score=5,
    criteria="""Correctness (1-5) - the factual accuracy of the answer.

- 1: Incorrect. The answer contains significant factual errors.
- 2: Mostly incorrect. The answer contains some correct information but major errors.
- 3: Partially correct. The answer is correct in some aspects but has notable errors.
- 4: Mostly correct. The answer is largely accurate with minor errors.
- 5: Completely correct. The answer is factually accurate throughout.""",
    evaluation_steps=[
        "Read the question and reference information if provided.",
        "Read the answer and identify factual claims.",
        "Verify the accuracy of these claims against known facts or reference.",
        "Assign a correctness score from 1 to 5."
    ],
    prompt_template="""You will be given a question and an answer to that question.

Your task is to rate the answer on one metric.

Please make sure you read and understand these instructions carefully.

Evaluation Criteria:

Correctness (1-5) - the factual accuracy of the answer.

- 1: Incorrect. The answer contains significant factual errors.
- 2: Mostly incorrect. The answer contains some correct information but major errors.
- 3: Partially correct. The answer is correct in some aspects but has notable errors.
- 4: Mostly correct. The answer is largely accurate with minor errors.
- 5: Completely correct. The answer is factually accurate throughout.

Evaluation Steps:

1. Read the question and reference information if provided.
2. Read the answer and identify factual claims.
3. Verify the accuracy of these claims against known facts or reference.
4. Assign a correctness score from 1 to 5.


Example:

Question:

{{Question}}

Answer:

{{Answer}}


Evaluation Form (scores ONLY):

- Correctness (1-5):"""
)


# Collection of all default dimensions
DEFAULT_SUMMARIZATION_DIMENSIONS = [FLUENCY, COHERENCE, CONSISTENCY, RELEVANCE]
DEFAULT_QA_DIMENSIONS = [HELPFULNESS, CORRECTNESS]
ALL_DIMENSIONS = DEFAULT_SUMMARIZATION_DIMENSIONS + DEFAULT_QA_DIMENSIONS


def get_dimension_by_name(name: str) -> EvaluationDimension:
    """Get a dimension by name.
    
    Args:
        name: Dimension name (e.g., "fluency", "coherence")
        
    Returns:
        EvaluationDimension
        
    Raises:
        ValueError: If dimension not found
    """
    for dim in ALL_DIMENSIONS:
        if dim.name.lower() == name.lower():
            return dim
    raise ValueError(f"Dimension '{name}' not found. Available: {[d.name for d in ALL_DIMENSIONS]}")

