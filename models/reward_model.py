"""
Reward Model Module

Provides APIs for evaluating question-answer pairs using reward models.
Supports multiple languages (Arabic, English) and uses the best available reward models.
Can use both local models and API-based approaches.
"""

import json
import os
import httpx
import asyncio
import re
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from utils.logger import setup_app_logger

logger = setup_app_logger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for a reward model."""
    model_name: str
    model_id: str  # HuggingFace model ID or API model name
    language: str  # 'arabic', 'english', or 'multilingual'
    api_type: str  # 'huggingface', 'openai', 'groq', 'vllm', 'local'
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    description: str = ""


# Best reward models for different languages
REWARD_MODEL_CONFIGS = {
    "arabic": RewardModelConfig(
        model_name="Arabic Reward Model",
        model_id="nvidia/llama-embed-nemotron-8b",
        language="arabic",
        api_type="huggingface",
        description="Best reward model for Arabic language with multilingual support"
    ),
    "english": RewardModelConfig(
        model_name="English Reward Model",
        model_id="nvidia/llama-embed-nemotron-8b",
        language="english",
        api_type="huggingface",
        description="Best reward model for English language"
    ),
    "multilingual": RewardModelConfig(
        model_name="Multilingual Reward Model",
        model_id="nvidia/llama-embed-nemotron-8b",
        language="multilingual",
        api_type="huggingface",
        description="Supports both Arabic and English"
    ),
}


class RewardModelAPI:
    """
    API-based reward model interface for evaluating answers.
    
    Supports multiple backends: HuggingFace Inference API, local models via vLLM, 
    and custom endpoints.
    """
    
    def __init__(
        self,
        language: str = "multilingual",
        api_type: str = "huggingface",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        model_config: Optional[RewardModelConfig] = None,
    ):
        """
        Initialize the Reward Model API.
        
        Args:
            language: Language for the reward model ('arabic', 'english', 'multilingual')
            api_type: Type of API ('huggingface', 'openai', 'vllm', 'local')
            api_key: API key for the service (e.g., HuggingFace token)
            api_endpoint: Custom API endpoint URL
            model_config: Custom model configuration
        """
        self.language = language.lower()
        self.api_type = api_type.lower()
        self.api_key = api_key or os.environ.get("HF_API_TOKEN", "")
        self.api_endpoint = api_endpoint
        self.timeout = httpx.Timeout(30.0, connect=10.0)
        
        # Load model configuration
        if model_config:
            self.config = model_config
        else:
            self.config = REWARD_MODEL_CONFIGS.get(
                self.language, 
                REWARD_MODEL_CONFIGS["multilingual"]
            )
        
        logger.info(
            f"Initialized RewardModelAPI for language: {self.language}, "
            f"using {self.api_type} API"
        )
    
    async def score_answer(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> float:
        """
        Score a single answer using the reward model.
        
        Args:
            question: The question asked
            answer: The answer to evaluate
            context: Optional context/source material
            
        Returns:
            A reward score between -1 and 1 (or 0 and 1 depending on model)
        """
        if self.api_type == "huggingface":
            return await self._score_with_huggingface(question, answer, context)
        elif self.api_type == "vllm":
            return await self._score_with_vllm(question, answer, context)
        elif self.api_type == "openai":
            return await self._score_with_openai(question, answer, context)
        elif self.api_type == "local":
            return await self._score_with_local(question, answer, context)
        else:
            logger.warning(f"Unknown API type: {self.api_type}. Returning neutral score.")
            return 0.5
    
    def score_answer_sync(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> float:
        """
        Synchronous wrapper for score_answer.
        
        Args:
            question: The question asked
            answer: The answer to evaluate
            context: Optional context/source material
            
        Returns:
            A reward score between -1 and 1 (or 0 and 1 depending on model)
        """
        try:
            return asyncio.run(self.score_answer(question, answer, context))
        except Exception as e:
            logger.error(f"Error scoring answer: {e}")
            return 0.0
    
    async def score_batch(
        self,
        qa_pairs: List[Dict[str, str]],
    ) -> List[float]:
        """
        Score multiple Q&A pairs in batch.
        
        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys
            
        Returns:
            List of reward scores
        """
        scores = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            context = qa.get("context")
            score = await self.score_answer(question, answer, context)
            scores.append(score)
        return scores
    
    def score_batch_sync(
        self,
        qa_pairs: List[Dict[str, str]],
    ) -> List[float]:
        """
        Synchronous wrapper for score_batch.
        
        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys
            
        Returns:
            List of reward scores
        """
        try:
            return asyncio.run(self.score_batch(qa_pairs))
        except Exception as e:
            logger.error(f"Error scoring batch: {e}")
            return [0.0] * len(qa_pairs)
    
    async def _score_with_huggingface(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> float:
        """Score using HuggingFace Inference API."""
        if not self.api_key:
            logger.warning("HuggingFace API key not provided. Using fallback scoring.")
            return 0.5
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                # Create input for the reward model
                # Format: [question, answer]
                input_text = f"{question} {answer}"
                
                payload = {
                    "inputs": input_text,
                }
                
                url = f"https://api-inference.huggingface.co/models/{self.config.model_id}"
                
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Extract score from response
                    if result and isinstance(result, list) and len(result) > 0:
                        # HuggingFace typically returns [{"score": value}]
                        score = result[0].get("score", 0.5)
                        return float(score)
                    else:
                        logger.warning(f"Unexpected HuggingFace API response format: {result}")
                        return 0.5
                        
                except httpx.HTTPStatusError as e:
                    logger.error(f"HuggingFace API HTTP error: {e.status_code} - {e.response.text}")
                    return 0.0
                except httpx.TimeoutException as e:
                    logger.error(f"HuggingFace API timeout: {e}")
                    return 0.0
                    
        except httpx.HTTPError as e:
            logger.error(f"HuggingFace API error: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error scoring with HuggingFace: {e}")
            return 0.0
    
    async def _score_with_vllm(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> float:
        """Score using vLLM inference endpoint with retry logic.

        This method supports two common vLLM server modes:
        - Completions/OpenAI-compatible endpoints ("completions")
        - Score-only mode (--task score) which exposes a /score endpoint

        It will try completions endpoints first and if the server returns an
        error indicating the model does not support the Completions API it will
        automatically try the /v1/score or /score endpoints and attempt to
        parse a numeric score from a variety of possible response formats.
        """
        if not self.api_endpoint:
            logger.warning("vLLM endpoint not provided.")
            return 0.5

        base = self.api_endpoint.rstrip('/')

        # endpoints to try in order: prefer /score but also try common variants if 404 occurs
        endpoints_to_try = [f"{base}/score", f"{base}/v1/score", f"{base}/score/"]

        max_retries = 3
        retry_delay = 1.0

        def _parse_score_from_result(res) -> Optional[float]:
            # Try many common response shapes for a numeric score
            try:
                if isinstance(res, dict):
                    # vLLM /score endpoint returns: {'data': [{'score': 0.64}]}
                    if isinstance(res.get('data'), list) and len(res.get('data')) > 0:
                        first_data = res.get('data')[0]
                        if isinstance(first_data, dict) and isinstance(first_data.get('score'), (int, float)):
                            return float(first_data.get('score'))
                    
                    # direct numeric fields
                    if isinstance(res.get('score'), (int, float)):
                        return float(res.get('score'))
                    if isinstance(res.get('scores'), list) and len(res.get('scores')) > 0:
                        first = res.get('scores')[0]
                        if isinstance(first, (int, float)):
                            return float(first)
                        if isinstance(first, dict) and isinstance(first.get('score'), (int, float)):
                            return float(first.get('score'))

                    if isinstance(res.get('predictions'), list) and len(res.get('predictions')) > 0:
                        p0 = res.get('predictions')[0]
                        if isinstance(p0, (int, float)):
                            return float(p0)
                        if isinstance(p0, dict):
                            for k in ('score', 'probability'):
                                if isinstance(p0.get(k), (int, float)):
                                    return float(p0.get(k))

                    if isinstance(res.get('choices'), list) and len(res.get('choices')) > 0:
                        choice = res.get('choices')[0]
                        # Try text field like completions
                        text = choice.get('text') or (choice.get('message') or {}).get('content')
                        if isinstance(text, str):
                            try:
                                return float(text.strip())
                            except Exception:
                                pass

                # Fallback: search for a 0..1 floating number in the JSON string
                s = json.dumps(res)
                m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", s)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return None
            except Exception:
                return None

            return None

        for attempt in range(max_retries):
            for endpoint in endpoints_to_try:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        prompt = f"Question: {question}\nAnswer: {answer}\n\nRate the answer quality (0-1):"

                        # Build the score-endpoint payload expected by vLLM --task score.
                        # vLLM /score endpoint expects: model, text_1, text_2, and optional fields.
                        # NOTE: do not include additionalProp1 or other unknown fields.
                        payload = {
                            "model": self.config.model_id,
                            "text_1": [question],
                            "text_2": [answer],
                        }

                        try:
                            logger.info(f"vLLM scoring: POST {endpoint} (attempt {attempt + 1}/{max_retries}) with model {self.config.model_id}")
                            response = await client.post(endpoint, json=payload)
                        except httpx.RequestError as e:
                            logger.info(f"Request error to {endpoint}: {e}")
                            continue

                        # If non-2xx, log body to help debug and try retry
                        if response.status_code >= 400:
                            body_text = None
                            try:
                                body_text = response.text
                            except Exception:
                                body_text = '<unreadable>'
                            logger.info(f"vLLM {endpoint} returned status {response.status_code}: {body_text[:200]}")
                            # try to parse any score-like info from body
                            try:
                                result = response.json()
                                logger.info(f"vLLM reward model response: {result}")
                                parsed = _parse_score_from_result(result)
                                if parsed is not None:
                                    return max(0.0, min(1.0, parsed))
                            except Exception:
                                pass
                            # will retry next cycle
                            continue

                        # Success; parse JSON
                        try:
                            result = response.json()
                        except Exception as e:
                            logger.debug(f"Failed to parse JSON from vLLM {endpoint}: {e}")
                            continue

                        logger.info(f"vLLM /score response (status 200): {result}")

                        # If the server returned an explicit error, log it and stop.
                        if isinstance(result, dict) and result.get('error'):
                            logger.error(f"vLLM returned error while scoring: {result.get('error')}")
                            return 0.0

                        # Try to parse numeric score from result
                        parsed = _parse_score_from_result(result)
                        if parsed is not None:
                            logger.info(f"vLLM score parsed: {parsed:.4f}")
                            return max(0.0, min(1.0, parsed))

                        logger.warning(f"Could not parse numeric score from vLLM /score response: {result}")
                        # nothing usable, will retry next cycle
                        continue

                except Exception as e:
                    logger.debug(f"Unexpected error calling vLLM endpoint {endpoint}: {e}")
                    continue

            # Backoff between retry cycles
            if attempt < max_retries - 1:
                logger.debug(f"vLLM scoring retry cycle {attempt + 1}/{max_retries} failed for all endpoints; sleeping {retry_delay}s before retrying...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

        logger.error(f"Error scoring with vLLM: Failed after {max_retries} attempts across endpoints")
        return 0.0
    
    async def _score_with_openai(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> float:
        """Score using OpenAI/Groq-compatible API."""
        try:
            import openai
        except ImportError:
            logger.warning("openai package not installed. Using fallback scoring.")
            return 0.5
        
        try:
            client = openai.AsyncOpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
            )
            
            prompt = f"""Rate the quality of this answer on a scale of 0 to 1.

Question: {question}

Answer: {answer}

Provide only a number between 0 and 1 as your response."""
            
            response = await client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            
            # Validate response
            if not response or not response.choices or len(response.choices) == 0:
                logger.warning(f"Invalid OpenAI response: {response}")
                return 0.5
            
            text = response.choices[0].message.content.strip()
            try:
                score = float(text)
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse score from OpenAI response: {text}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error scoring with OpenAI: {e}")
            return 0.0
    
    async def _score_with_local(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None,
    ) -> float:
        """Score using a local model (requires transformers library)."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            logger.warning("transformers not installed. Using fallback scoring.")
            return 0.5
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_id
            ).to(device)
            
            # Prepare input
            text = f"{question} {answer}"
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Normalize scores
                scores = torch.softmax(logits, dim=-1)
                # Assuming label 1 is positive
                score = float(scores[0][1].cpu().numpy())
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring with local model: {e}")
            return 0.0


class RewardModelEvaluator:
    """
    Evaluates synthetic QA pairs using reward models and filters based on quality.
    """
    
    def __init__(
        self,
        language: str = "multilingual",
        reward_threshold: float = 0.5,
        api_type: str = "huggingface",
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            language: Language for reward model
            reward_threshold: Minimum reward score to keep QA pair (0-1)
            api_type: Type of API to use
            api_key: API key for the service
            api_endpoint: Custom API endpoint
        """
        self.reward_model = RewardModelAPI(
            language=language,
            api_type=api_type,
            api_key=api_key,
            api_endpoint=api_endpoint,
        )
        self.reward_threshold = reward_threshold
        self.language = language
        logger.info(f"Initialized RewardModelEvaluator with threshold: {reward_threshold}")
    
    def evaluate_qa_pairs(
        self,
        qa_pairs: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, Union[str, float]]], List[float]]:
        """
        Evaluate QA pairs and return filtered results with scores.
        
        Args:
            qa_pairs: List of Q&A pairs to evaluate
            
        Returns:
            Tuple of (filtered_qa_pairs, all_scores)
        """
        try:
            scores = self.reward_model.score_batch_sync(qa_pairs)
        except Exception as e:
            logger.error(f"Error evaluating QA pairs: {e}")
            return qa_pairs, [0.5] * len(qa_pairs)
        
        filtered_pairs = []
        for qa, score in zip(qa_pairs, scores):
            if score >= self.reward_threshold:
                filtered_pairs.append({
                    **qa,
                    "reward_score": float(score)
                })
            else:
                logger.debug(
                    f"Filtering out QA pair due to low reward score ({score:.3f}): "
                    f"Q: {qa.get('question', '')[:50]}..."
                )
        
        logger.info(
            f"Evaluated {len(qa_pairs)} QA pairs. Kept {len(filtered_pairs)} "
            f"with reward >= {self.reward_threshold}"
        )
        
        return filtered_pairs, scores
    
    def score_single_answer(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Score a single answer.
        
        Args:
            question: The question
            answer: The answer to score
            
        Returns:
            Reward score
        """
        return self.reward_model.score_answer_sync(question, answer)


def create_reward_evaluator(
    language: str = "multilingual",
    reward_threshold: float = 0.5,
    api_type: str = "huggingface",
    api_key: Optional[str] = None,
    api_endpoint: Optional[str] = None,
) -> RewardModelEvaluator:
    """
    Factory function to create a reward model evaluator.
    
    Args:
        language: Language for reward model ('arabic', 'english', 'multilingual')
        reward_threshold: Minimum reward score to keep QA pair
        api_type: Type of API ('huggingface', 'openai', 'vllm', 'local')
        api_key: API key
        api_endpoint: Custom API endpoint
        
    Returns:
        RewardModelEvaluator instance
    """
    return RewardModelEvaluator(
        language=language,
        reward_threshold=reward_threshold,
        api_type=api_type,
        api_key=api_key,
        api_endpoint=api_endpoint,
    )


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize reward model
        reward_model = RewardModelAPI(language="multilingual", api_type="huggingface")
        
        # Score a single answer
        question = "What is 2+2?"
        answer = "4"
        score = await reward_model.score_answer(question, answer)
        print(f"Score for '{answer}': {score}")
        
        # Score multiple QA pairs
        qa_pairs = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2+2?", "answer": "5"},
        ]
        scores = await reward_model.score_batch(qa_pairs)
        print(f"Scores: {scores}")
    
    asyncio.run(main())
