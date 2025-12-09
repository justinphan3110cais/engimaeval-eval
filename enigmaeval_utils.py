"""
Consolidated utilities for EnigmaEval evaluation.
Contains all necessary functions, types, and constants.
"""

import os
import re
import pickle
from pathlib import Path
from pydantic import BaseModel
import datasets


# =================== SCHEMA ===================

class AnnotatedPuzzle(BaseModel):
    puzzle_id: str
    puzzle_source: str
    puzzle_text: str | None
    base64_images: list[str]
    answer: str
    prev_answers: str | None


# =================== CONSTANTS ===================

SOURCES = {
    "pp": "PuzzledPint",
    "lde": "Labor Day Extravaganza",
    "mit": "MIT Mystery Hunt",
    "crums": "CRUMS",
    "cc": "Cryptic Crossword",
    "potluck": "Puzzle Potluck",
    "cs50": "CS50x",
    "gm": "GMPuzzles",
}

SPLITS = {
    "normal": [
        "PuzzledPint",
        "Cryptic Crossword",
        "Puzzle Potluck",
        "CRUMS",
        "CS50x",
    ],
    "hard": [
        "MIT Mystery Hunt",
        "Labor Day Extravaganza",
        "GMPuzzles",
    ],
    "pdf": [
        "CRUMS",
        "Cryptic Crossword",
        "PuzzledPint",
        "Labor Day Extravaganza",
        "CS50x",
        "GMPuzzles",
    ],
    "html": ["MIT Mystery Hunt", "Puzzle Potluck"],
    "all": list(SOURCES.values()),
}


# =================== DATA LOADING ===================


# =================== DATA UTILITIES ===================

def fetch_puzzles(args):
    """
    Fetch puzzles from local pickle file based on arguments.
    
    The dataset should be exported using export_dataset.py first.
    Default location: data/enigmaeval.pkl
    """
    if args.split not in SOURCES and args.split not in SPLITS:
        raise ValueError(f"Invalid split: {args.split}")
    elif args.split in SOURCES:
        puzzle_sources = [SOURCES[args.split]]
    else:
        puzzle_sources = SPLITS[args.split]

    # Load dataset from pickle file
    dataset_path = getattr(args, 'dataset_path', 'data/enigmaeval.pkl')
    print(f"Loading dataset from: {dataset_path}")
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
        )
    
    with open(dataset_path, 'rb') as f:
        hf_dataset = pickle.load(f)
    
    # Shuffle with seed for reproducibility
    hf_dataset = hf_dataset.shuffle(seed=42)
    hf_dataset = hf_dataset.filter(lambda example: example["category"] in puzzle_sources)

    puzzles = []
    for row in hf_dataset:
        category = row["category"]
        puzzle_text = row["question"]  # The question text
        prev_answers = row["prev_answers"]
        
        # Handle meta puzzles
        if args.exclude_meta and "meta" in category.lower():
            continue

        # Handle images - the dataset has images as base64 strings (already cleaned)
        imgs = []
        if row["images"] is not None and str(row["images"]).strip():
            if args.text_only:
                continue  # Skip puzzles with images if text_only is requested
            
            # Images are already properly cleaned in the dataset
            imgs = [row["images"]]  # Single image as clean base64 string

        puzzles.append(
            AnnotatedPuzzle(
                puzzle_id=row["id"],
                puzzle_source=category,
                puzzle_text=puzzle_text,
                base64_images=imgs,
                answer=row["answer"],
                prev_answers=prev_answers,
            )
        )

    return puzzles


# =================== PROMPT UTILITIES ===================

def get_answer_type(answer: str):
    """Determine the answer type and format based on the answer string."""
    if "," in answer:
        if answer.count(",") == 1:
            answer_type = "pair of words/phrases"
            answer_format = "answer1, answer2"
        else:
            answer_type = "comma-separated list of words/phrases"
            answer_format = "answer1, answer2, ..."
    else:
        answer_type = "word or short phrase"
        answer_format = "answer"
    
    answer_format = f"<answer>{answer_format}</answer>"
    return answer_type, answer_format


class PromptMaker:
    """Creates prompts for different puzzle types."""
    
    def __init__(self, data_dir: str):
        """Initialize with prompt template directory."""
        data_dir = Path(data_dir)
        self.standard_tips = "\n" + (data_dir / "standard_tips.txt").read_text() + "\n---\n"
        self.imageless_tips = "\n" + (data_dir / "imageless_tips.txt").read_text() + "\n---\n"
        self.mit_tips = "\n" + (data_dir / "mit_tips.txt").read_text() + "\n---\n"
        self.plagiarism_prompt = "\n" + (data_dir / "plagiarism.txt").read_text() + "\n---\n"

    def get_prompt_pdf(self, answer: str, prev_answers: list[str] | None = None):
        """Generate prompt for PDF puzzles."""
        answer_type, answer_format = get_answer_type(answer)
        if prev_answers:
            preamble = f" This puzzle uses at least some of the following answers from other puzzles to arrive at its own solution: {prev_answers}."
        else:
            preamble = ""
        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}.
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Here are some tips for solving puzzles of this type:
{self.standard_tips}
"""

        user_prompt = f"""Your task is to solve the following puzzle, originally in PDF format and attached here as images.{preamble}
Write out a step-by-step solution to the puzzle. At the end of your solution, write your answer in the following format:
{answer_format}"""

        return user_prompt, system_prompt

    def get_prompt_standard_transcribed(
        self,
        puzzle_statement: str,
        answer: str,
        prev_answers: list[str] | None = None,
    ):
        """Generate prompt for standard transcribed puzzles."""
        answer_type, answer_format = get_answer_type(answer)

        has_image = "[image_1]" in puzzle_statement
        if has_image:
            preamble = (
                " The attached images are presented in the order they are referenced in the text."
            )
        else:
            preamble = ""
        if prev_answers:
            preamble = f" This puzzle uses at least some of the following answers from other puzzles to arrive at its own solution: {prev_answers}.{preamble}"

        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}. 
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Here are some tips for solving puzzles of this type:
{self.standard_tips}
"""
        user_prompt = f"""Your task is to solve the following puzzle.{preamble}

Puzzle:
{puzzle_statement}

Write out a step-by-step solution to the puzzle. At the end of your solution, write your answer in the following format:
{answer_format}
"""
        return user_prompt, system_prompt

    def get_prompt_nocheating(
        self,
        puzzle_statement: str,
        answer: str,
        prev_answers: list[str] | None = None,
    ):
        """Generate no-cheating prompt for puzzles."""
        answer_type, answer_format = get_answer_type(answer)

        has_image = "[image_1]" in puzzle_statement
        if has_image:
            preamble = (
                " The attached images are presented in the order they are referenced in the text."
            )
        else:
            preamble = ""
        if prev_answers:
            preamble = f" This puzzle uses at least some of the following answers from other puzzles to arrive at its own solution: {prev_answers}.{preamble}"

        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}. 
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Do not search the internet for the answer to the puzzle. You must not consult any web forums or puzzle-solving websites. If you have seen this puzzle before, do not use that information to solve it.
Here are some tips for solving puzzles of this type:
{self.standard_tips}
"""
        user_prompt = f"""Your task is to solve the following puzzle.{preamble}

Puzzle:
{puzzle_statement}

Write out an original step-by-step solution to the puzzle. Each step of the solution should be based on your own reasoning. At the end of your solution, write your answer in the following format:
{answer_format}
"""
        return user_prompt, system_prompt

    def get_prompt_mit_transcribed(
        self, puzzle_statement: str, answer: str, include_general_tips: bool = True
    ):
        """Generate prompt for MIT Mystery Hunt puzzles."""
        answer_type, answer_format = get_answer_type(answer)

        has_image = "[image_1]" in puzzle_statement
        if has_image:
            preamble = (
                " The attached images are presented in the order they are referenced in the text."
            )
        else:
            preamble = ""

        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}. 
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Here are some tips for solving puzzles of this type:
{self.mit_tips}
"""
        user_prompt = f"""Your task is to solve the following puzzle.{preamble}

Puzzle:
{puzzle_statement}

Write out a step-by-step solution to the puzzle. At the end of your solution, write your answer in the following format:
{answer_format}
"""
        return user_prompt, system_prompt

    def get_prompt_plagiarism(self, model_response: str):
        """Generate plagiarism detection prompt."""
        return self.plagiarism_prompt.format(model_response=model_response)
