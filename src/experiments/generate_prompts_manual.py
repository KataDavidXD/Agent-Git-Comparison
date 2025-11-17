"""
Manually crafted prompts with MAXIMUM differentiation between CoT and Few-shot.

Strategy:
- Few-shot: ONLY examples, minimal instructions, pure imitation
- CoT: Explicit reasoning steps, "show your work", force logic display
"""

import json
from pathlib import Path


def create_introduction_prompts():
    """Create Introduction prompts with maximum difference."""
    
    # CoT: Force explicit reasoning at each step
    cot_prompt = {
        "id": "introduction_cot_amplified",
        "style": "cot",
        "system": """You are writing the Introduction section for a literature review.

IMPORTANT: You MUST show your step-by-step reasoning explicitly.

Step 1: ANALYZE WHY this topic matters
→ Think: What problem does this solve? Write your reasoning.

Step 2: IDENTIFY current research state  
→ Think: What has been done? Write your analysis.

Step 3: FIND research gaps
→ Think: What's missing? Write your identification process.

Step 4: DEFINE review objective
→ Think: What will we synthesize? Write your definition.

Step 5: STRUCTURE the organization
→ Think: How to organize? Write your structure plan.

For EACH step, write "Reasoning:" followed by your thought process, THEN write the actual content.
Make your logical connections EXPLICIT using "therefore", "because", "this leads to".""",
        "user": "Write the Introduction section (200-300 words) based on these papers. SHOW YOUR REASONING for each step:\n\n{abstracts_text}"
    }
    
    # Few-shot: ONLY examples, almost no instructions
    few_shot_prompt = {
        "id": "introduction_fewshot_amplified", 
        "style": "few_shot",
        "system": """Write an Introduction following these examples EXACTLY:

Example 1:
"Quantum computing represents a paradigm shift in computational capability, promising to solve problems intractable for classical systems. Recent research has advanced error correction codes, quantum algorithms, and hardware implementations. This review examines 25 foundational papers, organizing findings into three domains: theoretical frameworks, algorithmic innovations, and practical applications in cryptography and optimization."

Example 2:
"Autonomous vehicles have emerged as a transformative technology at the intersection of artificial intelligence and transportation. Current work addresses perception systems, decision-making algorithms, and safety validation. This review synthesizes 30 key studies, structuring analysis around sensor fusion techniques, planning architectures, and real-world deployment challenges."

Example 3:
"Federated learning enables privacy-preserving machine learning across distributed devices without centralizing data. Recent advances focus on communication efficiency, model aggregation, and differential privacy. This review analyzes 20 seminal works, categorizing contributions into three themes: optimization methods, privacy mechanisms, and application scenarios in healthcare and finance."

Match this style EXACTLY.""",
        "user": "Write the Introduction (200-300 words):\n\n{abstracts_text}"
    }
    
    return [cot_prompt, few_shot_prompt]


def create_analysis_prompts():
    """Create Analysis prompts with maximum difference."""
    
    # CoT: Force multi-hop reasoning chain
    cot_prompt = {
        "id": "analysis_cot_amplified",
        "style": "cot",
        "system": """You are writing the Literature Analysis section.

CRITICAL: Show your COMPLETE reasoning chain for each step.

Step 1: EXTRACT methods from each paper
→ Reasoning: For each paper, identify the core technique. Write: "Paper X uses METHOD because..."

Step 2: IDENTIFY common patterns
→ Reasoning: Compare methods. Write: "Methods A and B share PATTERN because..."

Step 3: CLASSIFY into categories  
→ Reasoning: Group similar methods. Write: "I categorize these as GROUP because..."

Step 4: COMPARE trade-offs
→ Reasoning: Analyze pros/cons. Write: "Method X trades Y for Z because..."

Step 5: SYNTHESIZE trends
→ Reasoning: Connect insights. Write: "These trends show EVOLUTION because..."

Use phrases like "This suggests...", "Therefore...", "The connection is...", "This implies..." to make reasoning VISIBLE.""",
        "user": "Write the Literature Analysis (300-400 words). SHOW ALL REASONING:\n\n{abstracts_text}\n\nContext: {introduction}"
    }
    
    # Few-shot: Pure pattern imitation
    few_shot_prompt = {
        "id": "analysis_fewshot_amplified",
        "style": "few_shot",
        "system": """Write a Literature Analysis matching these examples:

Example 1:
"The surveyed works adopt three primary approaches. Reinforcement learning methods (Papers 1, 4, 7, 9) optimize policies through environmental interaction, achieving adaptability but requiring extensive training. Supervised learning approaches (Papers 2, 5, 8, 10) leverage labeled datasets for rapid deployment, though constrained by data availability. Hybrid architectures (Papers 3, 6, 11) combine both paradigms, balancing sample efficiency and generalization. Cross-paper comparison reveals reinforcement learning excels in dynamic environments while supervised methods dominate structured tasks. Papers 4 and 8 demonstrate that ensemble approaches outperform single-method baselines by 15-20%."

Example 2:
"Recent work divides into encoder-based (Papers 1-5), decoder-based (Papers 6-10), and encoder-decoder (Papers 11-15) architectures. Encoder models like BERT achieve superior understanding through bidirectional context but lack generative capacity. Decoder models such as GPT excel at generation through autoregressive training yet struggle with comprehension tasks. Encoder-decoder variants like T5 balance both capabilities at increased computational cost. Performance analysis indicates encoder models lead in classification while decoder models dominate generation, with encoder-decoder architectures showing versatility across diverse tasks."

Follow this EXACT format: categorize methods → describe each with paper citations → compare approaches → identify patterns.""",
        "user": "Write the Literature Analysis (300-400 words):\n\n{abstracts_text}\n\nContext: {introduction}"
    }
    
    return [cot_prompt, few_shot_prompt]


def create_discussion_prompts():
    """Create Discussion prompts with maximum difference."""
    
    # CoT: Force argumentative reasoning chain
    cot_prompt = {
        "id": "discussion_cot_amplified",
        "style": "cot",
        "system": """You are writing the Critical Discussion section.

MANDATORY: Build an EXPLICIT argument chain showing ALL reasoning.

Step 1: EVALUATE contributions
→ Reasoning: "Paper X contributes Y. This matters because..."

Step 2: IDENTIFY limitations  
→ Reasoning: "However, gap G exists. I conclude this because..."

Step 3: ANALYZE implications
→ Reasoning: "This limitation means Z. The logical connection is..."

Step 4: PROPOSE directions
→ Reasoning: "Therefore, future work should W. This follows because..."

Step 5: JUSTIFY importance
→ Reasoning: "Direction W addresses gap G because..."

Make EVERY logical leap explicit. Use: "Given that...", "It follows that...", "This reasoning leads to...", "The implication is...".""",
        "user": "Write the Critical Discussion (300-400 words). SHOW YOUR COMPLETE ARGUMENT CHAIN:\n\n{abstracts_text}\n\nPrevious:\n{introduction}\n{analysis}"
    }
    
    # Few-shot: Critical style imitation only
    few_shot_prompt = {
        "id": "discussion_fewshot_amplified",
        "style": "few_shot",
        "system": """Write a Critical Discussion matching these examples:

Example 1:
"While transformer architectures achieve state-of-the-art results, three critical limitations persist. First, quadratic attention complexity prohibits processing long sequences, restricting applications in genomics and legal document analysis. Second, positional encodings fail to capture hierarchical structure, limiting performance on structured data. Third, interpretability remains elusive, hindering deployment in high-stakes domains like healthcare. Future research should pursue: (1) linear-complexity attention mechanisms enabling longer contexts, (2) hierarchical position representations respecting document structure, and (3) attention-based explanability frameworks providing human-interpretable rationales for predictions."

Example 2:
"Despite remarkable progress in multimodal learning, fundamental challenges remain. Current methods require massive paired datasets, creating data bottlenecks for low-resource domains. Models exhibit brittle cross-modal alignment, failing when distributions shift between modalities. Computational demands restrict real-time applications. Research priorities include: (1) self-supervised techniques reducing supervision requirements, (2) robust alignment objectives maintaining performance under distribution shifts, and (3) efficient architectures enabling edge deployment through model compression and distillation."

Match this structure: acknowledge progress → identify 3 specific gaps → propose targeted solutions.""",
        "user": "Write the Critical Discussion (300-400 words):\n\n{abstracts_text}\n\nPrevious:\n{introduction}\n{analysis}"
    }
    
    return [cot_prompt, few_shot_prompt]


def main():
    """Generate and save manually crafted prompts."""
    print("="*80)
    print("GENERATING MANUALLY CRAFTED PROMPTS")
    print("Strategy: AMPLIFY differences between CoT and Few-shot")
    print("="*80)
    print("\nCoT: Force explicit reasoning, show thinking")
    print("Few-shot: Pure imitation, minimal instructions")
    print()
    
    output_dir = Path(__file__).parent
    
    # Introduction
    intro_prompts = create_introduction_prompts()
    with open(output_dir / "prompts_introduction.json", 'w', encoding='utf-8') as f:
        json.dump(intro_prompts, f, indent=2, ensure_ascii=False)
    print(f"✓ prompts_introduction.json (2 variations)")
    
    # Analysis
    analysis_prompts = create_analysis_prompts()
    with open(output_dir / "prompts_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_prompts, f, indent=2, ensure_ascii=False)
    print(f"✓ prompts_analysis.json (2 variations)")
    
    # Discussion
    discussion_prompts = create_discussion_prompts()
    with open(output_dir / "prompts_discussion.json", 'w', encoding='utf-8') as f:
        json.dump(discussion_prompts, f, indent=2, ensure_ascii=False)
    print(f"✓ prompts_discussion.json (2 variations)")
    
    print("\n" + "="*80)
    print("MANUAL PROMPTS COMPLETE!")
    print("="*80)
    print(f"\nTotal branches: 2 × 2 × 2 = 8")
    print("\nKey differences:")
    print("  CoT → Explicit reasoning steps, 'show your work'")
    print("  Few-shot → Pure pattern matching, minimal guidance")
    print("\nExpected: LARGER differences than before!")


if __name__ == "__main__":
    main()

