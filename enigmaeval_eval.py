#!/usr/bin/env python3
"""
Enigma evaluation script adapted for the leaderboard harness.
Integrates with shared LLM agents and models configuration.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
import re
import numpy as np
import yaml
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from shared.llm_agents import get_llm_agent_class
from enigmaeval_utils import (
    AnnotatedPuzzle, 
    SPLITS, 
    SOURCES, 
    fetch_puzzles, 
    PromptMaker
)

# Set up logging
logger = logging.getLogger(__name__)

def standardize(input_string: str) -> str:
    """Standardize answer strings for comparison."""
    if input_string is np.nan or input_string is None:
        return ""
    
    input_string = str(input_string).lower()
    if "," in input_string:
        return set([standardize(s) for s in input_string.split(",")])
    else:
        return re.sub(r"[^a-zA-Z0-9]", "", input_string)

def load_models_config(config_path: str) -> dict:
    """Load models configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_config(model_name: str, models_config: dict) -> tuple:
    """Get model configuration and return model_name and generation_config."""
    if model_name not in models_config:
        raise ValueError(f"Model {model_name} not found in models config")
    
    config = models_config[model_name]
    full_model_name = config['model']
    generation_config = config.get('generation_config', {})
    
    return full_model_name, generation_config

async def process_puzzle_batch(
    messages_list: list[list[dict]],
    puzzle_ids: list[str],
    gt_answers: list[str],
    llm_agent,
    max_concurrent: int = 32
) -> tuple[list[str], list[str], list[bool]]:
    """Process a batch of pre-built messages with the LLM agent."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_puzzle(i: int) -> tuple[int, str]:
        async with semaphore:
            messages = messages_list[i]
            puzzle_id = puzzle_ids[i]
            
            for retry_count in range(3):
                try:
                    # Get response from LLM agent
                    response = await llm_agent.async_completions(messages)
                    return (i, response.content if response.content else "")
                    
                except Exception as e:
                    logger.error(f"Error processing puzzle '{puzzle_id}' (attempt {retry_count+1}/3): {e}")
                    if retry_count == 2:  # Last attempt (0, 1, 2)
                        logger.error(f"Failed to get response for puzzle '{puzzle_id}' after all retries")
                        return (i, "")
                    # Continue to next retry
                        
    # Process all puzzles concurrently with progress bar
    tasks = [process_single_puzzle(i) for i in range(len(messages_list))]
    
    # Use as_completed for real-time progress with accuracy and cost
    responses = [""] * len(tasks)
    model_answers = [None] * len(tasks)
    is_corrects = [False] * len(tasks)
    correct = 0
    processed = 0
    
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
    for task in pbar:
        idx, response = await task
        responses[idx] = response
        
        # Extract and evaluate answer
        model_answer = extract_answer_from_response(response)
        model_answers[idx] = model_answer
        is_correct = evaluate_answer(gt_answers[idx], model_answer)
        is_corrects[idx] = is_correct
        
        correct += int(is_correct)
        processed += 1
        
        # Update progress bar with accuracy and cost
        accuracy = 100 * correct / processed
        cost = llm_agent.all_token_usage.cost
        pbar.set_postfix({
            "acc": f"{accuracy:.1f}%",
            "cost": f"${cost:.3f}"
        })
    
    return responses, model_answers, is_corrects

def extract_answer_from_response(response: str) -> str:
    """Extract the answer from the model response using XML tags."""
    if not isinstance(response, str):
        return None
    
    # Extract from <answer></answer> XML tags
    xml_match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    if xml_match:
        answer = xml_match.group(1).strip()
        return answer
    
    return None

def evaluate_answer(gt_answer: str, model_answer: str) -> bool:
    """Evaluate if the model answer matches the ground truth."""
    if model_answer is None:
        return False
    
    if "|" in gt_answer:
        # Multiple acceptable answers
        return standardize(model_answer) in [standardize(a) for a in gt_answer.split("|")]
    else:
        # Single answer
        return standardize(model_answer) == standardize(gt_answer)

async def grade_puzzles(
    puzzles: list[AnnotatedPuzzle],
    llm_agent,
    save_path: str,
    overwrite: bool = False,
    max_concurrent: int = 32,
    raw: bool = False,
    collated_pdf: bool = False,
    text_only: bool = False,
    exclude_meta: bool = False
):
    """Grade puzzles using the LLM agent."""
    
    # Initialize prompt maker
    prompt_templates_dir = Path(__file__).parent / "prompt_templates"
    prompt_maker = PromptMaker(str(prompt_templates_dir))
    
    # Prepare data for processing
    puzzle_ids = []
    gt_answers = []
    puzzle_sources = []
    messages_list = []
    
    # Check if we should skip some puzzles (if not overwriting)
    if not overwrite and os.path.exists(save_path):
        with open(save_path, 'r') as f:
            existing_data = json.load(f)
        processed_ids = set(item["puzzle_id"] for item in existing_data)
        puzzles = [p for p in puzzles if p.puzzle_id not in processed_ids]
        logger.info(f"Skipping {len(processed_ids)} already processed puzzles")
    
    # Process each puzzle to create messages
    for puzzle in puzzles:
        if exclude_meta and "meta" in puzzle.puzzle_id.lower():
            continue
            
        if text_only and puzzle.base64_images:
            continue
            
        # Generate prompts based on puzzle type
        if puzzle.puzzle_text is None:
            text_prompt, system_prompt = prompt_maker.get_prompt_pdf(
                puzzle.answer, prev_answers=puzzle.prev_answers
            )
        elif puzzle.puzzle_source == "MIT Mystery Hunt":
            text_prompt, system_prompt = prompt_maker.get_prompt_mit_transcribed(
                puzzle.puzzle_text, puzzle.answer
            )
        else:
            text_prompt, system_prompt = prompt_maker.get_prompt_standard_transcribed(
                puzzle.puzzle_text, puzzle.answer, prev_answers=puzzle.prev_answers
            )
        
        # Build messages directly
        messages = [{"role": "system", "content": system_prompt}]        
        # Build user message content
        content = []
        content.append({"type": "text", "text": text_prompt})
        
        # Add images if present
        if puzzle.base64_images:
            content.extend([
                {"type": "image_url", "image_url": {"url": img_b64}}
                for img_b64 in puzzle.base64_images
            ])
        
        messages.append({"role": "user", "content": content})
        
        # Store data for processing
        puzzle_ids.append(puzzle.puzzle_id)
        gt_answers.append(puzzle.answer)
        puzzle_sources.append(puzzle.puzzle_source)
        messages_list.append(messages)
    
    if not puzzle_ids:
        logger.info("No puzzles to process")
        return
    
    logger.info(f"Processing {len(puzzle_ids)} puzzles")
    
    # Get responses from LLM
    responses, model_answers, is_corrects = await process_puzzle_batch(
        messages_list, puzzle_ids, gt_answers, llm_agent, max_concurrent
    )
    
    # Create results data
    results_data = []
    
    for i in range(len(puzzle_ids)):
        results_data.append({
            "puzzle_id": puzzle_ids[i],
            "puzzle_source": puzzle_sources[i],
            "gt_answer": gt_answers[i],
            "model_response": responses[i],
            "model_answer": model_answers[i],
            "is_correct": is_corrects[i],
        })
    
    # Save results
    if overwrite or not os.path.exists(save_path):
        all_results = results_data
    else:
        # Append to existing results
        with open(save_path, 'r') as f:
            existing_data = json.load(f)
        all_results = existing_data + results_data
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Return results for main() to display
    return all_results

def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Enigma evaluation with shared LLM agents")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True, help="Model name from models config")
    parser.add_argument("--models_config", type=str, required=True, help="Path to models config YAML")
    
    # Dataset configuration
    parser.add_argument("--split", "-s", type=str, default="all", help="Which puzzle sets to evaluate")
    parser.add_argument("--num_puzzles", "-n", type=int, default=None, help="Only run on first n puzzles")
    
    # Output configuration
    parser.add_argument("--output_dir", "-d", type=str, required=True, help="Output directory")
    
    # Processing options
    parser.add_argument("--raw", action="store_true", help="Use raw PDF")
    parser.add_argument("--exclude_meta", action="store_true", help="Exclude meta-puzzles")
    parser.add_argument("--text_only", action="store_true", help="Only run on text puzzles")
    parser.add_argument("--collated_pdf", action="store_true", help="Use collated PDF")
    
    # Execution options
    parser.add_argument("--overwrite", "-o", action="store_true", 
                       help="Overwrite existing results, otherwise only rerun on puzzles with no answer")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Maximum concurrent requests")
    
    return parser

async def main():
    """Main evaluation function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Load models configuration
    models_config = load_models_config(args.models_config)
    full_model_name, generation_config = get_model_config(args.model, models_config)
    
    # Create LLM agent
    llm_agent = get_llm_agent_class(full_model_name, generation_config)
    logger.info(f"Using model: {full_model_name}")
    
    # Fetch puzzles
    puzzles = fetch_puzzles(args)
    logger.info(f"Loaded {len(puzzles)} puzzles")
    
    if args.num_puzzles is not None:
        puzzles = puzzles[:args.num_puzzles]
        logger.info(f"Limited to first {args.num_puzzles} puzzles")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create save path
    suffix = f"{'_raw' if args.raw else ''}{'_coll' if args.collated_pdf else ''}"
    save_path = os.path.join(args.output_dir, f"{args.model}{suffix}.json")
    
    # Run evaluation
    all_results = await grade_puzzles(
        puzzles=puzzles,
        llm_agent=llm_agent,
        save_path=save_path,
        overwrite=args.overwrite,
        max_concurrent=args.max_concurrent,
        raw=args.raw,
        collated_pdf=args.collated_pdf,
        text_only=args.text_only,
        exclude_meta=args.exclude_meta
    )
    
    # Print evaluation results
    if all_results:
        total_correct = sum(result["is_correct"] for result in all_results)
        total_count = len(all_results)
        accuracy = total_correct / total_count if total_count > 0 else 0
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Overall accuracy: {accuracy:.3f} ({total_correct}/{total_count})")
        
        # Print accuracy by source
        source_stats = {}
        for result in all_results:
            source = result["puzzle_source"]
            if source not in source_stats:
                source_stats[source] = {"correct": 0, "total": 0}
            source_stats[source]["total"] += 1
            if result["is_correct"]:
                source_stats[source]["correct"] += 1
        
        print("\nAccuracy by source:")
        for source, stats in sorted(source_stats.items()):
            source_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {source:<25}: {source_accuracy:.3f} ({stats['correct']:>3}/{stats['total']:<3})")
        print("=" * 80)
    
    logger.info(f"Results saved to: {save_path}")
    
    print("\n" + "=" * 80)
    print("ENIGMAEVAL EVALUATION COMPLETE!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
