def create_reward_function(self, reward_type="accuracy"):
        """
        Create reward function for RL training
        
        Args:
            reward_type: Type of reward ("accuracy", "length", "hybrid")
        
        Returns:
            Reward function
        """
        def accuracy_reward(question, answer, correct_answer):
            """Reward based on answer correctness"""
            answer_lower = answer.lower().strip()
            correct_lower = correct_answer.lower().strip()
            
            # Exact match: high reward
            if answer_lower == correct_lower:
                return 1.0
            
            # Partial match: medium reward
            answer_words = set(answer_lower.split())
            correct_words = set(correct_lower.split())
            
            if len(correct_words) == 0:
                return 0.0
            
            overlap = len(answer_words.intersection(correct_words))
            overlap_score = overlap / len(correct_words)
            
            # Penalize if too long or too short
            length_ratio = len(answer.split()) / max(len(correct_answer.split()), 1)
            length_penalty = 1.0 if 0.5 <= length_ratio <= 2.0 else 0.8
            
            return overlap_score * length_penalty
        
        def length_reward(question, answer, correct_answer):
            """Reward concise answers"""
            ideal_length = len(correct_answer.split())
            actual_length = len(answer.split())
            
            # Penalize very long or very short answers
            if actual_length < ideal_length * 0.5:
                return 0.3  # Too short
            elif actual_length > ideal_length * 2.0:
                return 0.5  # Too long
            else:
                return 0.9  # Good length
        
        def hybrid_reward(question, answer, correct_answer):
            """Combine accuracy and length rewards"""
            acc_reward = accuracy_reward(question, answer, correct_answer)
            len_reward = length_reward(question, answer, correct_answer)
            
            # Weighted combination
            return 0.7 * acc_reward + 0.3 * len_reward
        
        if reward_type == "accuracy":
            return accuracy_reward
        elif reward_type == "length":
            return length_reward
        else:
            return hybrid_reward