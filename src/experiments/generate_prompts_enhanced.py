"""
Enhanced prompt generation with AMPLIFIED differences between CoT and Few-shot.

This version explicitly designs prompts to maximize observable differences:
- Few-shot: Emphasize style imitation, format consistency
- CoT: Emphasize reasoning chains, step-by-step logic
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Clean sys.path and add ONLY current project to FRONT
project_root = Path(__file__).parent.parent.parent.parent
current_src = str(project_root / "src")

# Remove any old AgentGit paths
sys.path = [p for p in sys.path if 'AgentGit_Paper' not in p and 'Desktop' not in p]

# Insert current project at the VERY FRONT
sys.path.insert(0, current_src)

print(f"Using src path: {current_src}")

from generators import SimplePromptGenerator

# Load .env from project root
load_dotenv(dotenv_path=project_root / ".env")


def generate_introduction_prompts_enhanced():
    """Generate Introduction prompts with AMPLIFIED differences."""
    generator = SimplePromptGenerator()
    
    # CoT version - emphasize REASONING process
    cot_draft = """
You are generating the Introduction section for a literature review on {topic}.

THINK STEP BY STEP:
Step 1: Analyze WHY this research area is important (what problem does it solve?)
Step 2: Identify the CURRENT STATE of research (what has been done?)
Step 3: Determine what GAPS exist in current understanding
Step 4: Define the OBJECTIVE of this review (what will we synthesize?)
Step 5: Preview the LOGICAL STRUCTURE of the review

Based on {num_papers} papers, follow these reasoning steps to write an Introduction (200-300 words).
Show your reasoning process clearly in the output.
"""
    
    # Few-shot version - emphasize STYLE IMITATION with concrete examples
    few_shot_draft = """
You are generating the Introduction section for a literature review on {topic}.

Follow the EXACT STYLE AND FORMAT of these example introductions:

EXAMPLE 1 (Transformer Architecture Review):
"Transformer architectures have revolutionized natural language processing since their introduction in 2017. Current research explores attention mechanisms, scalability, and efficiency improvements. This review synthesizes recent advances in transformer design, analyzing 15 key papers. We organize our analysis into three themes: architectural innovations, training strategies, and application domains."

EXAMPLE 2 (Reinforcement Learning Review):
"Reinforcement learning has emerged as a cornerstone of modern AI, enabling agents to learn optimal behaviors through environmental interaction. Recent work focuses on sample efficiency, multi-agent systems, and real-world deployment. This review examines 20 seminal papers, categorizing approaches by learning paradigm, reward structure, and scalability considerations."

IMITATE THE STYLE: opening with impact statement, current research focus, review scope, and organization preview.

Based on {num_papers} papers, write an Introduction (200-300 words) matching this style EXACTLY.
"""
    
    print("[Introduction] Generating ENHANCED CoT and Few-shot variations...")
    
    prompts = []
    
    # Generate CoT version
    cot_variations = generator.generate_styled_prompts(
        draft_prompt=cot_draft,
        role="system"
    )
    cot_prompt = next((v for v in cot_variations if v.style.value == "cot"), None)
    if cot_prompt:
        prompts.append({
            "id": "introduction_chain_of_thought_enhanced",
            "style": "cot",
            "system": cot_prompt.content,
            "user": "Write the Introduction section based on these papers:\n\n{abstracts_text}"
        })
    
    # Generate Few-shot version
    few_shot_variations = generator.generate_styled_prompts(
        draft_prompt=few_shot_draft,
        role="system"
    )
    few_shot_prompt = next((v for v in few_shot_variations if v.style.value == "few_shot"), None)
    if few_shot_prompt:
        prompts.append({
            "id": "introduction_few_shot_enhanced",
            "style": "few_shot",
            "system": few_shot_prompt.content,
            "user": "Write the Introduction section based on these papers:\n\n{abstracts_text}"
        })
    
    return prompts


def generate_analysis_prompts_enhanced():
    """Generate Analysis prompts with AMPLIFIED differences."""
    generator = SimplePromptGenerator()
    
    # CoT version - emphasize MULTI-HOP REASONING
    cot_draft = """
You are generating the Literature Analysis section for a literature review on {topic}.

THINK STEP BY STEP - MULTI-HOP REASONING:
Step 1: EXTRACT core methods from each paper (what techniques are used?)
Step 2: IDENTIFY common themes (what patterns emerge across papers?)
Step 3: CLASSIFY methods into coherent categories (how do we group them?)
Step 4: COMPARE strengths and weaknesses (what are trade-offs?)
Step 5: SYNTHESIZE insights (what trends connect different approaches?)
Step 6: IDENTIFY evolutionary patterns (how has the field progressed?)

Based on {num_papers} papers, follow this reasoning chain to write a Literature Analysis (300-400 words).
Make your logical connections EXPLICIT.
"""
    
    # Few-shot version - emphasize FORMAT AND STRUCTURE with examples
    few_shot_draft = """
You are generating the Literature Analysis section for a literature review on {topic}.

Follow the EXACT FORMAT of these example analyses:

EXAMPLE 1 (Neural Architecture Search):
"Recent papers propose three main approaches. First, gradient-based methods (Papers 1, 4, 7) optimize architecture parameters directly, offering efficiency but limited search spaces. Second, evolutionary algorithms (Papers 2, 5, 8) explore diverse designs but require substantial compute. Third, predictor-based approaches (Papers 3, 6, 9) estimate performance without training, balancing speed and accuracy. Comparing these approaches reveals a fundamental trade-off between search efficiency and architecture diversity. Papers 4 and 8 demonstrate that hybrid methods combining multiple paradigms achieve superior results."

EXAMPLE 2 (Few-shot Learning):
"The surveyed work divides into meta-learning (Papers 1-5), metric learning (Papers 6-10), and data augmentation (Papers 11-15) strategies. Meta-learning approaches like MAML achieve fast adaptation through optimized initialization. Metric learning methods such as Prototypical Networks leverage distance-based classification. Data augmentation techniques synthesize training examples to bridge the few-shot gap. Performance comparisons indicate meta-learning excels in cross-domain scenarios while metric learning performs better within-domain."

IMITATE: (1) Categorize methods, (2) Describe each category with paper citations, (3) Compare approaches, (4) Identify patterns.

Based on {num_papers} papers, write a Literature Analysis (300-400 words) matching this FORMAT exactly.
"""
    
    print("[Analysis] Generating ENHANCED CoT and Few-shot variations...")
    
    prompts = []
    
    cot_variations = generator.generate_styled_prompts(
        draft_prompt=cot_draft,
        role="system"
    )
    cot_prompt = next((v for v in cot_variations if v.style.value == "cot"), None)
    if cot_prompt:
        prompts.append({
            "id": "analysis_chain_of_thought_enhanced",
            "style": "cot",
            "system": cot_prompt.content,
            "user": "Write the Literature Analysis section based on these papers:\n\n{abstracts_text}\n\nIntroduction context:\n{introduction}"
        })
    
    few_shot_variations = generator.generate_styled_prompts(
        draft_prompt=few_shot_draft,
        role="system"
    )
    few_shot_prompt = next((v for v in few_shot_variations if v.style.value == "few_shot"), None)
    if few_shot_prompt:
        prompts.append({
            "id": "analysis_few_shot_enhanced",
            "style": "few_shot",
            "system": few_shot_prompt.content,
            "user": "Write the Literature Analysis section based on these papers:\n\n{abstracts_text}\n\nIntroduction context:\n{introduction}"
        })
    
    return prompts


def generate_discussion_prompts_enhanced():
    """Generate Discussion prompts with MAXIMUM difference amplification."""
    generator = SimplePromptGenerator()
    
    # CoT version - emphasize COMPLEX ARGUMENTATION CHAIN
    cot_draft = """
You are generating the Critical Discussion section for a literature review on {topic}.

THINK STEP BY STEP - BUILD AN ARGUMENT:
Step 1: EVALUATE what each paper contributes (individual merits?)
Step 2: SYNTHESIZE the collective contribution (overall progress?)
Step 3: IDENTIFY gaps (what's missing in current research?)
Step 4: ANALYZE limitations (methodological or conceptual weaknesses?)
Step 5: REASON about implications (what do limitations mean for the field?)
Step 6: PROPOSE future directions (how do we address these gaps?)
Step 7: JUSTIFY why these directions are important (logical connection to gaps)

Based on {num_papers} papers, follow this argumentative reasoning to write a Critical Discussion (300-400 words).
Show clear LOGICAL CONNECTIONS between evaluations, gaps, and proposals.
"""
    
    # Few-shot version - emphasize CRITICAL TONE AND STRUCTURE
    few_shot_draft = """
You are generating the Critical Discussion section for a literature review on {topic}.

Follow the EXACT TONE AND STRUCTURE of these examples:

EXAMPLE 1 (Computer Vision):
"While recent advances in self-supervised learning demonstrate impressive results, three critical gaps remain. First, current methods struggle with fine-grained visual distinctions, limiting applicability in medical and satellite imaging. Second, computational requirements prohibit deployment on edge devices, restricting real-world adoption. Third, theoretical understanding of why contrastive methods work remains incomplete, hindering principled improvements. These limitations suggest three research directions: developing hierarchical representations for multi-scale features, designing efficient architectures via neural architecture search, and establishing theoretical frameworks connecting contrastive objectives to downstream task performance."

EXAMPLE 2 (Natural Language Generation):
"Despite significant progress in large language models, fundamental challenges persist. The reliance on massive datasets raises concerns about data privacy and environmental impact. Models exhibit inconsistent factual accuracy, producing plausible but incorrect information. Controllability remains limited, with difficulty steering generation toward specific styles or constraints. Future work should prioritize: (1) efficient training methods reducing data and compute requirements, (2) retrieval-augmented architectures grounding generation in verified sources, and (3) fine-grained control mechanisms enabling precise output specifications."

IMITATE THE STRUCTURE: (1) Acknowledge progress, (2) Identify 3 specific gaps, (3) Explain implications, (4) Propose targeted future directions with justification.

Based on {num_papers} papers, write a Critical Discussion (300-400 words) matching this CRITICAL TONE exactly.
"""
    
    print("[Discussion] Generating ENHANCED CoT and Few-shot variations...")
    
    prompts = []
    
    cot_variations = generator.generate_styled_prompts(
        draft_prompt=cot_draft,
        role="system"
    )
    cot_prompt = next((v for v in cot_variations if v.style.value == "cot"), None)
    if cot_prompt:
        prompts.append({
            "id": "discussion_chain_of_thought_enhanced",
            "style": "cot",
            "system": cot_prompt.content,
            "user": "Write the Critical Discussion section based on these papers:\n\n{abstracts_text}\n\nIntroduction:\n{introduction}\n\nAnalysis:\n{analysis}"
        })
    
    few_shot_variations = generator.generate_styled_prompts(
        draft_prompt=few_shot_draft,
        role="system"
    )
    few_shot_prompt = next((v for v in few_shot_variations if v.style.value == "few_shot"), None)
    if few_shot_prompt:
        prompts.append({
            "id": "discussion_few_shot_enhanced",
            "style": "few_shot",
            "system": few_shot_prompt.content,
            "user": "Write the Critical Discussion section based on these papers:\n\n{abstracts_text}\n\nIntroduction:\n{introduction}\n\nAnalysis:\n{analysis}"
        })
    
    return prompts


def main():
    """Generate ENHANCED prompt variations with amplified differences."""
    print("="*80)
    print("GENERATING ENHANCED FEW-SHOT AND COT PROMPT VARIATIONS")
    print("Strategy: AMPLIFY differences to make them observable in output")
    print("="*80)
    
    output_dir = Path(__file__).parent
    
    try:
        # Generate Introduction prompts
        intro_prompts = generate_introduction_prompts_enhanced()
        intro_file = output_dir / "prompts_introduction.json"
        with open(intro_file, 'w', encoding='utf-8') as f:
            json.dump(intro_prompts, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {intro_file.name} ({len(intro_prompts)} variations)")
        print(f"        CoT: Emphasizes reasoning steps")
        print(f"        Few-shot: Emphasizes style imitation")
        
        # Generate Analysis prompts
        analysis_prompts = generate_analysis_prompts_enhanced()
        analysis_file = output_dir / "prompts_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_prompts, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {analysis_file.name} ({len(analysis_prompts)} variations)")
        print(f"        CoT: Emphasizes multi-hop reasoning")
        print(f"        Few-shot: Emphasizes format consistency")
        
        # Generate Discussion prompts
        discussion_prompts = generate_discussion_prompts_enhanced()
        discussion_file = output_dir / "prompts_discussion.json"
        with open(discussion_file, 'w', encoding='utf-8') as f:
            json.dump(discussion_prompts, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {discussion_file.name} ({len(discussion_prompts)} variations)")
        print(f"        CoT: Emphasizes argumentation chain")
        print(f"        Few-shot: Emphasizes critical tone")
        
        print("\n" + "="*80)
        print("ENHANCED GENERATION COMPLETE!")
        print("="*80)
        print(f"\nTotal branches: {len(intro_prompts)} × {len(analysis_prompts)} × {len(discussion_prompts)} = {len(intro_prompts) * len(analysis_prompts) * len(discussion_prompts)}")
        print("\nExpected differences:")
        print("  Few-shot → More consistent format, polished style")
        print("  CoT → More explicit reasoning, logical connections")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

