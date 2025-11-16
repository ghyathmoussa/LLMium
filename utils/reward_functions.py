


"""
Reward Functions for Answer Quality Evaluation

Provides various reward functions for evaluating the quality of answers
in different contexts. These can be used with reward models or as standalone
quality metrics.
"""

import re
import string
from typing import List, Dict, Any
from utils.logger import setup_app_logger

logger = setup_app_logger(__name__)


def format_reward_func(completions, **kwargs):
    """
    Check if the completion follows a specific format with thinking and answer sections.
    
    Expected format: <think>...</think><answer>...</answer>
    
    Args:
        completions: List of lists of dicts with 'content' field
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (0.0 or 1.0)
    """
    rewards = []
    for comp_list in completions:  # completions is a list of lists of dicts
        # Assuming conversational format [{ "role": "assistant", "content": "..." }]
        if comp_list and isinstance(comp_list, list) and len(comp_list) > 0 and "content" in comp_list[0]:
            content = comp_list[0]["content"]
            # Check for proper format
            has_think = "<think>" in content and "</think>" in content
            has_answer = "<answer>" in content and "</answer>" in content
            
            if has_think and has_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)  # Penalize malformed completions
    return rewards


def accuracy_reward_func(completions, solution, **kwargs):
    """
    Check if the answer in the completion matches the provided solution.
    
    Args:
        completions: List of lists of dicts with 'content' field
        solution: List of expected solutions
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (0.0 or 1.0)
    """
    rewards = []
    if not solution or len(solution) != len(completions):
        return [0.0] * len(completions)

    for i, comp_list in enumerate(completions):
        if comp_list and isinstance(comp_list, list) and len(comp_list) > 0 and "content" in comp_list[0]:
            generated_content = comp_list[0]["content"].strip()
            expected_solution = str(solution[i]).strip()
            
            # Exact match
            if generated_content == expected_solution:
                rewards.append(1.0)
            # Partial match (case-insensitive)
            elif generated_content.lower() == expected_solution.lower():
                rewards.append(0.8)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def answer_length_reward_func(completions, min_length=10, max_length=500, **kwargs):
    """
    Reward answers based on their length being within acceptable bounds.
    
    Args:
        completions: List of lists of dicts with 'content' field
        min_length: Minimum acceptable length in characters
        max_length: Maximum acceptable length in characters
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (0.0 to 1.0)
    """
    rewards = []
    for comp_list in completions:
        if comp_list and isinstance(comp_list, list) and len(comp_list) > 0 and "content" in comp_list[0]:
            content = comp_list[0]["content"]
            length = len(content)
            
            if min_length <= length <= max_length:
                rewards.append(1.0)
            elif length < min_length:
                # Penalty proportional to how short it is
                rewards.append(max(0.0, length / min_length * 0.5))
            else:
                # Penalty proportional to how long it is
                excess = length - max_length
                rewards.append(max(0.0, 1.0 - (excess / max_length) * 0.5))
        else:
            rewards.append(0.0)
    return rewards


def answer_relevance_reward_func(completions, question, **kwargs):
    """
    Reward answers based on how relevant they are to the question.
    Uses simple keyword overlap as a heuristic.
    
    Args:
        completions: List of lists of dicts with 'content' field
        question: The question being answered
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (0.0 to 1.0)
    """
    rewards = []
    
    # Extract key terms from question (simple approach: remove stopwords)
    question_lower = question.lower()
    # Remove common stopwords (English and Arabic)
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'have', 'has',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'what', 'when', 'where', 'why', 'how', 'which', 'who',
        'في', 'من', 'إلى', 'هذا', 'ذلك', 'التي', 'الذي', 'على', 'عن'
    }
    
    question_words = set(
        word.strip(string.punctuation)
        for word in question_lower.split()
        if word.strip(string.punctuation) not in stopwords and len(word.strip(string.punctuation)) > 2
    )
    
    for comp_list in completions:
        if comp_list and isinstance(comp_list, list) and len(comp_list) > 0 and "content" in comp_list[0]:
            content = comp_list[0]["content"].lower()
            content_words = set(
                word.strip(string.punctuation)
                for word in content.split()
                if len(word.strip(string.punctuation)) > 2
            )
            
            if question_words:
                overlap = len(question_words & content_words) / len(question_words)
                rewards.append(min(1.0, overlap))
            else:
                rewards.append(0.5)
        else:
            rewards.append(0.0)
    
    return rewards


def non_empty_reward_func(completions, **kwargs):
    """
    Simple reward function that gives 1.0 for non-empty answers, 0.0 otherwise.
    
    Args:
        completions: List of lists of dicts with 'content' field
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (0.0 or 1.0)
    """
    rewards = []
    for comp_list in completions:
        if (comp_list and isinstance(comp_list, list) and len(comp_list) > 0 
            and "content" in comp_list[0] and comp_list[0]["content"].strip()):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def language_quality_reward_func(completions, language='arabic', **kwargs):
    """
    Reward answers based on language quality indicators.
    
    Args:
        completions: List of lists of dicts with 'content' field
        language: Language to check ('arabic' or 'english')
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (0.0 to 1.0)
    """
    rewards = []
    
    for comp_list in completions:
        if comp_list and isinstance(comp_list, list) and len(comp_list) > 0 and "content" in comp_list[0]:
            content = comp_list[0]["content"]
            
            score = 0.5  # Neutral base score
            
            # Check for proper spacing
            if "  " in content:  # Double spaces suggest poor formatting
                score -= 0.1
            
            # Check for punctuation at the end
            if content and content[-1] in '.!?؟۔':
                score += 0.2
            
            # Check for reasonable word count
            words = content.split()
            if len(words) > 3:
                score += 0.1
            
            if language.lower() == 'arabic':
                # Check for Arabic characters
                arabic_pattern = re.compile(r'[\u0600-\u06FF]')
                if arabic_pattern.search(content):
                    score += 0.1
            elif language.lower() == 'english':
                # Check for English characters
                if any(c.isalpha() and ord(c) < 128 for c in content):
                    score += 0.1
            
            rewards.append(min(1.0, max(0.0, score)))
        else:
            rewards.append(0.0)
    
    return rewards


def composite_reward_func(completions, weights=None, **kwargs):
    """
    Combine multiple reward functions with weighted averaging.
    
    Args:
        completions: List of lists of dicts with 'content' field
        weights: Dict mapping function names to weights
        **kwargs: Additional arguments passed to individual functions
        
    Returns:
        List of composite reward scores (0.0 to 1.0)
    """
    if weights is None:
        weights = {
            'format': 0.2,
            'length': 0.2,
            'relevance': 0.3,
            'language_quality': 0.3,
        }
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    composite_scores = [0.0] * len(completions)
    
    if 'format' in weights and weights['format'] > 0:
        scores = format_reward_func(completions, **kwargs)
        for i, score in enumerate(scores):
            composite_scores[i] += score * weights['format']
    
    if 'length' in weights and weights['length'] > 0:
        scores = answer_length_reward_func(completions, **kwargs)
        for i, score in enumerate(scores):
            composite_scores[i] += score * weights['length']
    
    if 'relevance' in weights and weights['relevance'] > 0:
        scores = answer_relevance_reward_func(completions, **kwargs)
        for i, score in enumerate(scores):
            composite_scores[i] += score * weights['relevance']
    
    if 'language_quality' in weights and weights['language_quality'] > 0:
        scores = language_quality_reward_func(completions, **kwargs)
        for i, score in enumerate(scores):
            composite_scores[i] += score * weights['language_quality']
    
    return composite_scores