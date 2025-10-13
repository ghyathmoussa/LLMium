"""
Advanced Utilities for Battle AI Agent
- Data augmentation
- Advanced evaluation metrics
- Performance monitoring
- Dataset quality analysis
"""

import json
import random
from typing import List, Dict, Tuple
import re
from collections import defaultdict

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmenter:
    """Augment training data to improve model robustness"""
    
    def __init__(self):
        self.paraphrase_templates = {
            "what_is": [
                "What is {entity}?",
                "Can you explain what {entity} is?",
                "Define {entity}.",
                "Tell me about {entity}.",
                "Explain {entity} to me.",
            ],
            "how_does": [
                "How does {entity} work?",
                "Explain how {entity} works.",
                "Can you describe how {entity} functions?",
            ],
            "why": [
                "Why {question}?",
                "What's the reason for {question}?",
                "Explain why {question}.",
            ]
        }
    
    def paraphrase_question(self, question: str, num_variants: int = 3) -> List[str]:
        """Generate paraphrased versions of a question"""
        variants = [question]
        
        # Simple paraphrasing by adding question starters
        starters = [
            "Could you tell me: ",
            "I'd like to know: ",
            "Please explain: ",
            "",
        ]
        
        for starter in starters[:num_variants]:
            variants.append(starter + question)
        
        return variants[:num_variants + 1]
    
    def add_question_variations(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Add question variations to dataset"""
        augmented_data = []
        
        for qa in qa_pairs:
            # Original pair
            augmented_data.append(qa)
            
            # Generate variants
            variants = self.paraphrase_question(qa['question'], num_variants=2)
            
            for variant in variants[1:]:  # Skip original
                augmented_data.append({
                    'question': variant,
                    'answer': qa['answer']
                })
        
        return augmented_data
    
    def add_negative_examples(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Add examples of what NOT to do (for DPO training)"""
        dpo_data = []
        
        bad_answer_templates = [
            "I don't know.",
            "{answer_start}...",  # Incomplete
            "The answer is complicated.",
            lambda a: a[:len(a)//2],  # Truncated
        ]
        
        for qa in qa_pairs:
            # Create preference pair
            chosen = qa['answer']
            
            # Create a bad answer
            if len(chosen.split()) > 5:
                rejected = chosen.split('.')[0] + '.'  # Only first sentence
            else:
                rejected = "I'm not sure about that."
            
            dpo_data.append({
                'question': qa['question'],
                'chosen': chosen,
                'rejected': rejected
            })
        
        return dpo_data
    
    def balance_dataset(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Balance dataset by question types and difficulty"""
        
        # Categorize questions
        categories = defaultdict(list)
        
        for qa in qa_pairs:
            q_lower = qa['question'].lower()
            
            if q_lower.startswith(('what', 'which')):
                categories['definition'].append(qa)
            elif q_lower.startswith(('how', 'explain')):
                categories['explanation'].append(qa)
            elif q_lower.startswith(('why', 'what causes')):
                categories['reasoning'].append(qa)
            elif any(op in q_lower for op in ['+', '-', '*', '/', 'calculate']):
                categories['math'].append(qa)
            else:
                categories['other'].append(qa)
        
        # Find minimum category size
        min_size = min(len(items) for items in categories.values() if items)
        
        # Balance by sampling
        balanced = []
        for category, items in categories.items():
            if items:
                sampled = random.sample(items, min(len(items), min_size * 2))
                balanced.extend(sampled)
        
        return balanced


# ============================================================================
# ADVANCED EVALUATION METRICS
# ============================================================================

class AdvancedEvaluator:
    """Sophisticated evaluation metrics for Q&A"""
    
    def __init__(self):
        self.metrics = {}
    
    def exact_match(self, prediction: str, reference: str) -> float:
        """Exact match score"""
        pred_normalized = prediction.lower().strip()
        ref_normalized = reference.lower().strip()
        return 1.0 if pred_normalized == ref_normalized else 0.0
    
    def token_overlap_f1(self, prediction: str, reference: str) -> float:
        """F1 score based on token overlap"""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 0.0
        
        # Calculate precision and recall
        common = pred_tokens.intersection(ref_tokens)
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def contains_answer(self, prediction: str, reference: str) -> float:
        """Check if prediction contains the answer"""
        pred_lower = prediction.lower()
        ref_lower = reference.lower()
        
        # Check if reference is substring of prediction
        if ref_lower in pred_lower:
            return 1.0
        
        # Check if key terms from reference appear in prediction
        ref_words = set(ref_lower.split())
        pred_words = set(pred_lower.split())
        
        overlap = len(ref_words.intersection(pred_words))
        return overlap / len(ref_words) if ref_words else 0.0
    
    def length_penalty(self, prediction: str, reference: str) -> float:
        """Penalize answers that are too long or too short"""
        pred_len = len(prediction.split())
        ref_len = len(reference.split())
        
        if ref_len == 0:
            return 0.0
        
        ratio = pred_len / ref_len
        
        # Ideal ratio is close to 1.0
        if 0.5 <= ratio <= 2.0:
            return 1.0 - abs(1.0 - ratio) * 0.5
        elif ratio < 0.5:
            return 0.3  # Too short
        else:
            return 0.5  # Too long
    
    def semantic_similarity(self, prediction: str, reference: str) -> float:
        """Simple semantic similarity (can be enhanced with embeddings)"""
        # Extract key entities/numbers
        pred_entities = self._extract_entities(prediction)
        ref_entities = self._extract_entities(reference)
        
        if not ref_entities:
            return self.token_overlap_f1(prediction, reference)
        
        # Check entity overlap
        common_entities = pred_entities.intersection(ref_entities)
        entity_score = len(common_entities) / len(ref_entities)
        
        # Combine with token overlap
        token_score = self.token_overlap_f1(prediction, reference)
        
        return 0.6 * token_score + 0.4 * entity_score
    
    def _extract_entities(self, text: str) -> set:
        """Extract numbers, capitalized words, and special terms"""
        entities = set()
        
        # Numbers
        numbers = re.findall(r'\d+', text)
        entities.update(numbers)
        
        # Capitalized words (potential proper nouns)
        words = text.split()
        capitalized = [w for w in words if w and w[0].isupper()]
        entities.update(capitalized)
        
        return entities
    
    def comprehensive_score(
        self,
        prediction: str,
        reference: str,
        weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive score with multiple metrics"""
        
        if weights is None:
            weights = {
                'exact_match': 0.2,
                'token_f1': 0.3,
                'contains_answer': 0.2,
                'semantic': 0.2,
                'length': 0.1
            }
        
        scores = {
            'exact_match': self.exact_match(prediction, reference),
            'token_f1': self.token_overlap_f1(prediction, reference),
            'contains_answer': self.contains_answer(prediction, reference),
            'semantic': self.semantic_similarity(prediction, reference),
            'length': self.length_penalty(prediction, reference),
        }
        
        # Calculate weighted average
        overall = sum(scores[k] * weights[k] for k in scores)
        scores['overall'] = overall
        
        return scores


# ============================================================================
# DATASET QUALITY ANALYZER
# ============================================================================

class DatasetAnalyzer:
    """Analyze dataset quality and characteristics"""
    
    def analyze_dataset(self, qa_pairs: List[Dict]) -> Dict:
        """Comprehensive dataset analysis"""
        
        analysis = {
            'total_pairs': len(qa_pairs),
            'question_stats': self._analyze_questions([qa['question'] for qa in qa_pairs]),
            'answer_stats': self._analyze_answers([qa['answer'] for qa in qa_pairs]),
            'diversity': self._analyze_diversity(qa_pairs),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_questions(self, questions: List[str]) -> Dict:
        """Analyze question characteristics"""
        lengths = [len(q.split()) for q in questions]
        
        # Question types
        types = defaultdict(int)
        for q in questions:
            q_lower = q.lower()
            if q_lower.startswith('what'):
                types['what'] += 1
            elif q_lower.startswith('how'):
                types['how'] += 1
            elif q_lower.startswith('why'):
                types['why'] += 1
            elif q_lower.startswith('who'):
                types['who'] += 1
            elif q_lower.startswith('when'):
                types['when'] += 1
            elif q_lower.startswith('where'):
                types['where'] += 1
            else:
                types['other'] += 1
        
        return {
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'types': dict(types),
        }
    
    def _analyze_answers(self, answers: List[str]) -> Dict:
        """Analyze answer characteristics"""
        lengths = [len(a.split()) for a in answers]
        
        return {
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'very_short': sum(1 for l in lengths if l < 5),
            'short': sum(1 for l in lengths if 5 <= l < 20),
            'medium': sum(1 for l in lengths if 20 <= l < 50),
            'long': sum(1 for l in lengths if l >= 50),
        }
    
    def _analyze_diversity(self, qa_pairs: List[Dict]) -> Dict:
        """Analyze dataset diversity"""
        
        # Unique questions and answers
        unique_questions = len(set(qa['question'] for qa in qa_pairs))
        unique_answers = len(set(qa['answer'] for qa in qa_pairs))
        
        # Vocabulary size
        all_words = set()
        for qa in qa_pairs:
            all_words.update(qa['question'].lower().split())
            all_words.update(qa['answer'].lower().split())
        
        return {
            'unique_questions': unique_questions,
            'unique_answers': unique_answers,
            'vocabulary_size': len(all_words),
            'question_diversity': unique_questions / len(qa_pairs) if qa_pairs else 0,
            'answer_diversity': unique_answers / len(qa_pairs) if qa_pairs else 0,
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check dataset size
        if analysis['total_pairs'] < 500:
            recommendations.append("⚠️  Dataset is small. Consider collecting more Q&A pairs (target: 1000+)")
        
        # Check question diversity
        if analysis['diversity']['question_diversity'] < 0.8:
            recommendations.append("⚠️  Many duplicate questions. Increase question diversity.")
        
        # Check answer length distribution
        answer_stats = analysis['answer_stats']
        if answer_stats['very_short'] / analysis['total_pairs'] > 0.5:
            recommendations.append("⚠️  Many very short answers. Add more detailed responses.")
        
        # Check question type balance
        types = analysis['question_stats']['types']
        max_type_ratio = max(types.values()) / sum(types.values()) if types else 0
        if max_type_ratio > 0.6:
            recommendations.append("⚠️  Question types are imbalanced. Add variety (what, how, why, etc.)")
        
        if not recommendations:
            recommendations.append("✓ Dataset quality looks good!")
        
        return recommendations


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Monitor and log agent performance during training/inference"""
    
    def __init__(self):
        self.metrics_history = []
    
    def log_inference(self, question: str, answer: str, latency: float):
        """Log inference metrics"""
        self.metrics_history.append({
            'question': question,
            'answer': answer,
            'latency': latency,
            'answer_length': len(answer.split()),
        })
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        latencies = [m['latency'] for m in self.metrics_history]
        answer_lengths = [m['answer_length'] for m in self.metrics_history]
        
        return {
            'total_inferences': len(self.metrics_history),
            'avg_latency': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
            'throughput': len(self.metrics_history) / sum(latencies) if sum(latencies) > 0 else 0,
        }
    
    def save_report(self, filepath: str):
        """Save performance report"""
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_metrics': self.metrics_history
            }, f, indent=2)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Advanced Utilities - Demo")
    print("="*70)
    print()
    
    # Sample data
    sample_qa = [
        {"question": "What is Python?", "answer": "Python is a programming language."},
        {"question": "How does photosynthesis work?", "answer": "Photosynthesis converts light into energy."},
        {"question": "What is 5 + 3?", "answer": "8"},
    ]
    
    # 1. Data Augmentation
    print("1. DATA AUGMENTATION")
    print("-"*70)
    augmenter = DataAugmenter()
    augmented = augmenter.add_question_variations(sample_qa)
    print(f"Original pairs: {len(sample_qa)}")
    print(f"Augmented pairs: {len(augmented)}")
    print(f"Example variants:")
    for qa in augmented[:5]:
        print(f"  - {qa['question']}")
    print()
    
    # 2. Dataset Analysis
    print("2. DATASET ANALYSIS")
    print("-"*70)
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze_dataset(sample_qa)
    print(f"Total pairs: {analysis['total_pairs']}")
    print(f"Avg question length: {analysis['question_stats']['avg_length']:.1f} words")
    print(f"Avg answer length: {analysis['answer_stats']['avg_length']:.1f} words")
    print(f"Vocabulary size: {analysis['diversity']['vocabulary_size']}")
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
    print()
    
    # 3. Advanced Evaluation
    print("3. ADVANCED EVALUATION")
    print("-"*70)
    evaluator = AdvancedEvaluator()
    
    prediction = "Python is a high-level programming language"
    reference = "Python is a programming language"
    
    scores = evaluator.comprehensive_score(prediction, reference)
    print(f"Prediction: {prediction}")
    print(f"Reference: {reference}")
    print(f"\nScores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")
    print()
    
    print("="*70)
    print("✓ Utilities demo complete!")