import re
from typing import Callable, Dict, List
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

class RewardFunctions:
    """Collection of reward functions for training chat models"""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize sentence embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.reward_registry = {
            "accuracy": self.accuracy_reward,
            "length": self.length_reward,
            "hybrid": self.hybrid_reward,
            "coherence": self.coherence_reward,
            "relevance": self.relevance_reward,
            "helpfulness": self.helpfulness_reward,
            "format_quality": self.format_quality_reward,
            "safety": self.safety_reward,
            "instruction_following": self.instruction_following_reward,
            "factuality": self.factuality_reward,
            "engagement": self.engagement_reward,
            "multi_objective": self.multi_objective_reward
        }
    
    def get_reward_function(self, reward_type: str) -> Callable:
        """Get reward function by type"""
        return self.reward_registry.get(reward_type, self.hybrid_reward)
    
    def accuracy_reward(self, question: str, answer: str, correct_answer: str, **kwargs) -> float:
        """Reward based on answer correctness using cosine similarity"""
        answer_lower = answer.lower().strip()
        correct_lower = correct_answer.lower().strip()
        
        # Exact match gets perfect score
        if answer_lower == correct_lower:
            return 1.0
        
        # Handle empty answers
        if not answer_lower or not correct_lower:
            return 0.0
        
        # Compute sentence embeddings
        answer_embedding = self.embedding_model.encode(answer_lower, convert_to_tensor=False)
        correct_embedding = self.embedding_model.encode(correct_lower, convert_to_tensor=False)
        
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(answer_embedding, correct_embedding)
        
        # Ensure similarity is in valid range [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        
        # Apply length penalty for answers that are too short or too long
        length_ratio = len(answer.split()) / max(len(correct_answer.split()), 1)
        length_penalty = 1.0 if 0.5 <= length_ratio <= 2.0 else 0.9
        
        return similarity * length_penalty
    
    def length_reward(self, question: str, answer: str, correct_answer: str, **kwargs) -> float:
        """Reward concise answers"""
        ideal_length = len(correct_answer.split())
        actual_length = len(answer.split())
        
        if actual_length < ideal_length * 0.5:
            return 0.3
        elif actual_length > ideal_length * 2.0:
            return 0.5
        else:
            return 0.9
    
    def hybrid_reward(self, question: str, answer: str, correct_answer: str, **kwargs) -> float:
        """Combine accuracy and length rewards"""
        acc_reward = self.accuracy_reward(question, answer, correct_answer)
        len_reward = self.length_reward(question, answer, correct_answer)
        return 0.7 * acc_reward + 0.3 * len_reward
    
    def coherence_reward(self, question: str, answer: str, correct_answer: str = None, **kwargs) -> float:
        """
        Reward coherent, well-structured responses
        Checks for repetition, sentence structure, and logical flow
        """
        reward = 1.0
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        # Penalize excessive repetition
        words = answer.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # Too repetitive
                reward *= 0.6
        
        # Penalize very short sentences repeatedly
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        if avg_sentence_length < 3:
            reward *= 0.7
        
        # Check for broken sentences (no verb/structure)
        complete_sentences = sum(1 for s in sentences if len(s.split()) >= 3)
        completeness = complete_sentences / len(sentences)
        reward *= completeness
        
        return max(0.0, min(1.0, reward))
    
    def relevance_reward(self, question: str, answer: str, correct_answer: str = None, **kwargs) -> float:
        """
        Reward answers that are relevant to the question
        Uses keyword overlap and semantic similarity proxies
        """
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 
                      'when', 'where', 'who', 'which', 'do', 'does', 'did', 'can', 'could'}
        question_keywords = question_words - stop_words
        answer_keywords = answer_words - stop_words
        
        if len(question_keywords) == 0:
            return 0.8
        
        # Calculate keyword overlap
        overlap = len(question_keywords.intersection(answer_keywords))
        relevance_score = overlap / len(question_keywords)
        
        # Bonus if answer directly addresses question words
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who']
        has_starter = any(q in question.lower() for q in question_starters)
        
        if has_starter and len(answer.split()) > 5:
            relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def helpfulness_reward(self, question: str, answer: str, correct_answer: str = None, **kwargs) -> float:
        """
        Reward helpful, informative responses
        Considers detail, examples, and actionability
        """
        reward = 0.5  # Base score
        
        # Check for sufficient detail
        word_count = len(answer.split())
        if word_count >= 20:
            reward += 0.2
        elif word_count < 5:
            reward -= 0.3
        
        # Check for examples or explanations
        has_example = any(marker in answer.lower() for marker in 
                         ['for example', 'such as', 'like', 'e.g.', 'for instance'])
        if has_example:
            reward += 0.15
        
        # Check for structure (numbered lists, bullets)
        has_structure = bool(re.search(r'(\d+\.|•|\-|\*)\s', answer))
        if has_structure and word_count > 30:
            reward += 0.15
        
        # Penalize unhelpful responses
        unhelpful_phrases = ['i don\'t know', 'i cannot', 'i\'m not sure', 
                            'i don\'t have information', 'sorry, i can\'t']
        if any(phrase in answer.lower() for phrase in unhelpful_phrases):
            reward *= 0.5
        
        return max(0.0, min(1.0, reward))
    
    def format_quality_reward(self, question: str, answer: str, correct_answer: str = None, **kwargs) -> float:
        """
        Reward well-formatted responses
        Checks punctuation, capitalization, spacing
        """
        reward = 1.0
        
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 0:
            properly_capitalized = sum(1 for s in sentences if s and s[0].isupper())
            cap_ratio = properly_capitalized / len(sentences)
            reward *= cap_ratio
        
        # Check for proper punctuation
        if len(answer) > 10 and not re.search(r'[.!?]$', answer.strip()):
            reward *= 0.8
        
        # Penalize excessive caps
        if answer.isupper() and len(answer) > 20:
            reward *= 0.3
        
        # Penalize excessive punctuation
        punct_count = sum(1 for c in answer if c in '!?')
        if punct_count > len(answer.split()) * 0.3:
            reward *= 0.7
        
        return max(0.0, min(1.0, reward))
    
    def safety_reward(self, question: str, answer: str, correct_answer: str = None, **kwargs) -> float:
        """
        Reward safe, appropriate responses
        Penalizes toxic, harmful, or inappropriate content
        """
        answer_lower = answer.lower()
        
        # List of harmful patterns (expand based on your use case)
        harmful_patterns = [
            r'\b(kill|murder|hurt|harm)\s+\w+',
            r'\b(hate|racist|sexist)\b',
            r'\bexplicit\s+content\b',
            r'\billegal\s+\w+',
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, answer_lower):
                return 0.0
        
        # Check for refusal patterns (sometimes appropriate)
        refusal_patterns = ['i cannot', 'i won\'t', 'i can\'t help with that']
        has_refusal = any(phrase in answer_lower for phrase in refusal_patterns)
        
        # Context matters: if question seems harmful, refusal is good
        harmful_question = any(word in question.lower() 
                              for word in ['hack', 'illegal', 'harm', 'attack'])
        
        if harmful_question and has_refusal:
            return 1.0  # Good refusal
        elif harmful_question and not has_refusal:
            return 0.3  # Should have refused
        
        return 0.9  # Default safe
    
    def instruction_following_reward(self, question: str, answer: str, correct_answer: str = None, **kwargs) -> float:
        """
        Reward following specific instructions in the question
        Checks for format requirements, constraints, etc.
        """
        reward = 1.0
        question_lower = question.lower()
        
        # Check for format requirements
        format_checks = {
            'list': (lambda a: bool(re.search(r'(\d+\.|•|\-)\s', a)), 'list format'),
            'bullet': (lambda a: bool(re.search(r'(•|\-|\*)\s', a)), 'bullet points'),
            'short': (lambda a: len(a.split()) <= 50, 'brief response'),
            'detailed': (lambda a: len(a.split()) >= 50, 'detailed response'),
            'example': (lambda a: 'example' in a.lower() or 'for instance' in a.lower(), 'examples'),
        }
        
        for keyword, (check_func, desc) in format_checks.items():
            if keyword in question_lower:
                if check_func(answer):
                    reward += 0.1
                else:
                    reward *= 0.7
        
        # Check for "don't" instructions
        if 'don\'t' in question_lower or 'do not' in question_lower:
            # Extract what not to do (simplified)
            forbidden = re.findall(r'don\'t\s+(\w+)|do\s+not\s+(\w+)', question_lower)
            forbidden_words = [w for group in forbidden for w in group if w]
            
            if any(word in answer.lower() for word in forbidden_words):
                reward *= 0.5
        
        return max(0.0, min(1.0, reward))
    
    def factuality_reward(self, question: str, answer: str, correct_answer: str = None, 
                         known_facts: Dict = None, **kwargs) -> float:
        """
        Reward factually accurate responses
        Uses known facts database or correct_answer as reference
        """
        if known_facts is None and correct_answer is None:
            return 0.8  # Neutral score if no reference
        
        reward = 0.5
        
        # If correct_answer provided, use it
        if correct_answer:
            answer_lower = answer.lower()
            correct_lower = correct_answer.lower()
            
            # Check for key facts
            correct_words = set(correct_lower.split())
            answer_words = set(answer_lower.split())
            
            if len(correct_words) > 0:
                overlap = len(correct_words.intersection(answer_words))
                fact_score = overlap / len(correct_words)
                reward += 0.5 * fact_score
        
        # Penalize hedging language that might indicate uncertainty
        hedging = ['might', 'maybe', 'possibly', 'perhaps', 'could be', 'i think']
        hedge_count = sum(1 for h in hedging if h in answer.lower())
        
        if hedge_count > 2:
            reward *= 0.8
        
        return max(0.0, min(1.0, reward))
    
    def engagement_reward(self, question: str, answer: str, correct_answer: str = None, **kwargs) -> float:
        """
        Reward engaging, natural conversational responses
        Considers tone, variety, and interactivity
        """
        reward = 0.6
        
        # Check for conversational markers
        conversational = ['here', 'let me', 'i can', 'you can', 'this means', 'in other words']
        has_conversational = sum(1 for marker in conversational if marker in answer.lower())
        reward += min(0.2, has_conversational * 0.05)
        
        # Check for variety in sentence structure
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            if length_variance > 5:  # Good variety
                reward += 0.1
        
        # Penalize overly robotic responses
        robotic_starters = ['the answer is', 'the result is', 'the solution is']
        if any(answer.lower().startswith(starter) for starter in robotic_starters):
            reward *= 0.9
        
        return max(0.0, min(1.0, reward))
    
    def multi_objective_reward(self, question: str, answer: str, correct_answer: str = None,
                              weights: Dict[str, float] = None, **kwargs) -> float:
        """
        Combine multiple reward objectives with custom weights
        
        Args:
            weights: Dictionary mapping reward types to their weights
                    e.g., {'accuracy': 0.3, 'helpfulness': 0.2, ...}
        """
        if weights is None:
            # Default weights for general chat
            weights = {
                'accuracy': 0.25,
                'helpfulness': 0.20,
                'coherence': 0.15,
                'relevance': 0.15,
                'format_quality': 0.10,
                'safety': 0.10,
                'engagement': 0.05
            }
        
        total_reward = 0.0
        total_weight = 0.0
        
        for reward_type, weight in weights.items():
            if reward_type in self.reward_registry:
                reward_func = self.reward_registry[reward_type]
                reward_value = reward_func(question, answer, correct_answer, **kwargs)
                total_reward += weight * reward_value
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return total_reward / total_weight
        
        return 0.5
    
    def compute_reward(self, question: str, answer: str, correct_answer: str = None,
                      reward_type: str = "multi_objective", **kwargs) -> float:
        """
        Main interface to compute reward
        
        Args:
            question: Input question
            answer: Model's answer
            correct_answer: Ground truth answer (optional)
            reward_type: Type of reward to compute
            **kwargs: Additional arguments for specific reward functions
        
        Returns:
            Reward score between 0 and 1
        """
        reward_func = self.get_reward_function(reward_type)
        return reward_func(question, answer, correct_answer, **kwargs)


# Example usage
if __name__ == "__main__":
    rf = RewardFunctions()
    
    question = "What is the capital of France?"
    answer = "The capital of France is Paris. It's a beautiful city known for art and culture."
    correct_answer = "Paris"
    
    # Test different reward functions
    print(f"Accuracy Reward: {rf.compute_reward(question, answer, correct_answer, 'accuracy'):.3f}")
    print(f"Helpfulness Reward: {rf.compute_reward(question, answer, correct_answer, 'helpfulness'):.3f}")
    print(f"Coherence Reward: {rf.compute_reward(question, answer, correct_answer, 'coherence'):.3f}")
    print(f"Multi-objective Reward: {rf.compute_reward(question, answer, correct_answer, 'multi_objective'):.3f}")
    
    # Custom weights
    custom_weights = {
        'accuracy': 0.4,
        'helpfulness': 0.3,
        'coherence': 0.2,
        'engagement': 0.1
    }
    print(f"\nCustom Multi-objective: {rf.compute_reward(question, answer, correct_answer, 'multi_objective', weights=custom_weights):.3f}")