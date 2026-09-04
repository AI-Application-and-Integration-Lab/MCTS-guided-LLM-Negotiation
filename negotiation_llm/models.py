"""
Abstract model interface and implementations for LLM-based negotiation.
"""

import math
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(ABC):
    """Abstract base class for generation models."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response as string
        """
        pass

    def generate_batch(self, prompt: str, num_samples: int = 3, **kwargs) -> list[str]:
        """
        Generate multiple responses from the same prompt (batched generation).

        Inspired by GDPZero's batched generation for creating multiple realizations.
        Default implementation calls generate() multiple times, but can be overridden
        for more efficient batched inference.

        Args:
            prompt: Input text prompt
            num_samples: Number of responses to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated responses
        """
        return [self.generate(prompt, **kwargs) for _ in range(num_samples)]

    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass


class HuggingFaceModel(BaseModel):
    """Hugging Face model implementation for local inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-8B-Instruct",
        quantization: Optional[str] = "4bit",
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs
    ):
        """
        Initialize the Hugging Face model.

        Args:
            model_name: Model identifier from Hugging Face
            quantization: Quantization method ("4bit", "8bit", or None)
            device_map: Device mapping strategy
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            **kwargs: Additional model parameters
        """
        # Import torch and transformers here to avoid import errors when using MockModel
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch  # Store torch for use in other methods
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        # Configure quantization if specified
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        print(f"Loading model: {model_name}")
        if quantization:
            print(f"Using {quantization} quantization")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization is None else None,
            **kwargs
        )

        print("Model loaded successfully!")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Input text prompt
            **kwargs: Override generation parameters

        Returns:
            Generated response as string
        """
        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        if "Qwen3" in self.model_name:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:   
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Get generation parameters
        gen_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Generate
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_params
            )

        # Decode only the generated tokens (exclude input)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def generate_batch(self, prompt: str, num_samples: int = 3, **kwargs) -> list[str]:
        """
        Generate multiple responses efficiently using num_return_sequences.

        Optimized batched generation inspired by GDPZero's parallel generation.

        Args:
            prompt: Input text prompt
            num_samples: Number of responses to generate
            **kwargs: Override generation parameters

        Returns:
            List of generated responses
        """
        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        if "Qwen3" in self.model_name:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Get generation parameters with num_return_sequences
        gen_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": num_samples,  # Generate multiple sequences
        }

        # Generate
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_params
            )

        # Decode all generated sequences
        responses = []
        for output in outputs:
            generated_tokens = output[inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())

        return responses

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'torch') and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class VLLMModel(BaseModel):
    """
    vLLM model implementation for high-performance inference.

    vLLM provides:
    - 10-20x faster inference than HuggingFace
    - Efficient batched generation
    - PagedAttention for optimized memory usage
    - Continuous batching for higher throughput
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-8B-Instruct",
        tensor_parallel_size: int = -1,
        max_model_len: int = 5120,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        gpu_memory_utilization: float = 0.8,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        lora_modules: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize vLLM model.

        Args:
            model_name: Model identifier from Hugging Face
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            enable_lora: Enable multi-LoRA support
            max_lora_rank: Maximum LoRA rank
            lora_modules: Dict mapping lora_name -> lora_path for pre-loading adapters
            **kwargs: Additional vLLM parameters
        """
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            import torch
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest

        # LoRA configuration
        self.enable_lora = enable_lora
        self.lora_modules = lora_modules or {}
        self.lora_requests = {}  # Cache of LoRARequest objects

        if tensor_parallel_size == -1:
            gpu_count = torch.cuda.device_count()
            tensor_parallel_size = max(1, gpu_count)
            print(f"Auto-detected {gpu_count} GPU(s), using tensor_parallel_size={tensor_parallel_size}")

        print(f"Loading vLLM model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
        if self.enable_lora:
            print(f"LoRA enabled: max_rank={max_lora_rank}")
            if self.lora_modules:
                print(f"Pre-configured LoRA modules: {list(self.lora_modules.keys())}")

        # Filter out HuggingFace-specific parameters that vLLM doesn't support
        vllm_incompatible = {'device_map', 'quantization'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in vllm_incompatible}

        # Initialize vLLM with LoRA support if enabled
        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            **filtered_kwargs
        }

        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank

        self.llm = LLM(**llm_kwargs)

        # Get tokenizer for chat template
        self.tokenizer = self.llm.get_tokenizer()

        # Pre-create LoRA requests for configured modules
        for idx, (lora_name, lora_path) in enumerate(self.lora_modules.items(), start=1):
            self.lora_requests[lora_name] = LoRARequest(
                lora_name=lora_name,
                lora_int_id=idx,
                lora_path=lora_path
            )
            print(f"LoRA adapter '{lora_name}' configured from {lora_path}")

        print("vLLM model loaded successfully!")

    def generate(self, prompt: str, lora_adapter: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Input text prompt
            lora_adapter: Name of LoRA adapter to use (must be in lora_modules)
            **kwargs: Override generation parameters

        Returns:
            Generated response as string
        """
        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        if "Qwen3" in self.model_name:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Create sampling parameters
        sampling_params = self.SamplingParams(
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens)),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
        )

        # Get LoRA request if adapter specified
        lora_request = None
        if lora_adapter and lora_adapter in self.lora_requests:
            lora_request = self.lora_requests[lora_adapter]

        # Generate with optional LoRA adapter
        outputs = self.llm.generate([text], sampling_params, use_tqdm=False, lora_request=lora_request)
        response = outputs[0].outputs[0].text

        return response.strip()

    def generate_batch(self, prompt: str, num_samples: int = 3, lora_adapter: Optional[str] = None, **kwargs) -> list[str]:
        """
        Generate multiple responses efficiently using vLLM's batching.

        vLLM's continuous batching automatically handles multiple requests
        efficiently. This method generates multiple samples from the same prompt.

        Args:
            prompt: Input text prompt
            num_samples: Number of responses to generate
            lora_adapter: Name of LoRA adapter to use (must be in lora_modules)
            **kwargs: Override generation parameters

        Returns:
            List of generated responses
        """
        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        if "Qwen3" in self.model_name:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Create sampling parameters with n (number of samples)
        sampling_params = self.SamplingParams(
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens)),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            n=num_samples,  # Generate multiple samples
        )

        # Get LoRA request if adapter specified
        lora_request = None
        if lora_adapter and lora_adapter in self.lora_requests:
            lora_request = self.lora_requests[lora_adapter]

        # Generate - vLLM will efficiently generate all samples
        outputs = self.llm.generate([text], sampling_params, use_tqdm=False, lora_request=lora_request)

        # Extract all generated samples
        responses = [output.text.strip() for output in outputs[0].outputs]

        return responses

    def estimate_phrases_probability_batch(
        self,
        prompt: str,
        target_phrases: list[str],
        use_chat_template: bool = True,
        logprobs: int = 5,
        lora_adapter: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Estimate probabilities for multiple target phrases in a single batched LLM call.

        Much more efficient than calling estimate_phrase_probability multiple times,
        as vLLM processes all prompts in parallel.

        Args:
            prompt: Input text prompt (context)
            target_phrases: List of phrases whose probabilities we want to estimate
            use_chat_template: Whether to apply chat template to prompt
            logprobs: Number of top logprobs to return per token (for inspection)
            lora_adapter: Name of LoRA adapter to score under (must be in
                lora_modules). Omitting it scores under the base model, which
                is rarely what you want when a trained policy exists.

        Returns:
            Dictionary mapping each phrase to its probability result dict
        """
        # Prepare prompt with chat template if requested
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            if "Qwen3" in self.model_name:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            formatted_prompt = prompt

        # Build all extended prompts (prompt + each phrase)
        extended_prompts = [formatted_prompt + phrase for phrase in target_phrases]

        # Run vLLM ONCE on all extended prompts in parallel
        sampling_params = self.SamplingParams(
            max_tokens=1,  # We only need prompt logprobs
            prompt_logprobs=logprobs,
            temperature=0.0,
        )

        lora_request = None
        if lora_adapter and lora_adapter in self.lora_requests:
            lora_request = self.lora_requests[lora_adapter]

        outputs = self.llm.generate(
            extended_prompts, sampling_params, use_tqdm=False, lora_request=lora_request
        )

        # Tokenize base prompt once
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        prompt_token_count = len(prompt_tokens)

        # For each phrase, compute probability of its suffix
        results = {}
        for phrase, output in zip(target_phrases, outputs):
            prompt_logprobs_data = output.prompt_logprobs

            # Tokenize full prompt + phrase
            full_tokens = self.tokenizer.encode(formatted_prompt + phrase)
            phrase_tokens = full_tokens[prompt_token_count:]

            # Extract logprobs for phrase tokens
            token_logprobs = []
            token_texts = []

            for i, token_id in enumerate(phrase_tokens):
                logprob_idx = prompt_token_count + i

                if logprob_idx < len(prompt_logprobs_data) and prompt_logprobs_data[logprob_idx]:
                    token_logprob_dict = prompt_logprobs_data[logprob_idx]
                    if token_id in token_logprob_dict:
                        token_logprob = token_logprob_dict[token_id].logprob
                        token_logprobs.append(token_logprob)
                        token_texts.append(self.tokenizer.decode([token_id]))
                    else:
                        token_logprobs.append(-20.0)
                        token_texts.append(self.tokenizer.decode([token_id]))
                else:
                    token_logprobs.append(-20.0)
                    token_texts.append(self.tokenizer.decode([token_id]))

            # Calculate statistics
            if token_logprobs:
                import math
                total_logprob = sum(token_logprobs)
                avg_logprob = total_logprob / len(token_logprobs)
                try:
                    probability = math.exp(total_logprob)
                except OverflowError:
                    probability = 0.0
                try:
                    perplexity = math.exp(-avg_logprob)
                except OverflowError:
                    perplexity = float('inf')
            else:
                total_logprob = 0.0
                avg_logprob = 0.0
                probability = 0.0
                perplexity = float('inf')

            results[phrase] = {
                "total_logprob": total_logprob,
                "avg_logprob": avg_logprob,
                "probability": probability,
                "perplexity": perplexity,
                "num_tokens": len(token_logprobs),
                "tokens": phrase_tokens,
                "token_logprobs": token_logprobs,
                "token_texts": token_texts,
            }

        return results

    def estimate_phrase_probability(
        self,
        prompt: str,
        target_phrase: str,
        use_chat_template: bool = True,
        logprobs: int = 5,
        lora_adapter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Estimate the probability of a target phrase given a prompt.

        Uses vLLM's prompt logprobs feature to compute the log probability
        of generating the target phrase as the continuation of the prompt.

        Args:
            prompt: Input text prompt (context)
            target_phrase: The phrase whose probability we want to estimate
            use_chat_template: Whether to apply chat template to prompt
            logprobs: Number of top logprobs to return per token (for inspection)
            lora_adapter: Name of LoRA adapter to score under (must be in
                lora_modules). Omitting it scores under the base model.

        Returns:
            Dictionary containing:
            - total_logprob: Sum of log probabilities for all tokens in target_phrase
            - avg_logprob: Average log probability per token
            - probability: Approximate probability (exp of total_logprob)
            - perplexity: Perplexity of the target phrase
            - tokens: List of tokens in target_phrase
            - token_logprobs: List of log probabilities for each token
            - token_texts: List of decoded text for each token
        """
        # Prepare prompt with chat template if requested
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            if "Qwen3" in self.model_name:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            formatted_prompt = prompt

        # Combine prompt and target phrase to get logprobs
        # vLLM's prompt_logprobs parameter returns logprobs for the prompt tokens
        full_text = formatted_prompt + target_phrase

        # Create sampling parameters with logprobs enabled
        # We set max_tokens=1 since we only need the prompt logprobs
        sampling_params = self.SamplingParams(
            max_tokens=1,
            prompt_logprobs=logprobs,  # Get logprobs for prompt tokens
            temperature=0.0,  # Use greedy decoding for consistency
        )

        # Generate to get logprobs
        lora_request = None
        if lora_adapter and lora_adapter in self.lora_requests:
            lora_request = self.lora_requests[lora_adapter]

        outputs = self.llm.generate(
            [full_text], sampling_params, use_tqdm=False, lora_request=lora_request
        )

        # Extract prompt logprobs
        prompt_logprobs_data = outputs[0].prompt_logprobs

        # Tokenize the formatted prompt and target phrase separately
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        full_tokens = self.tokenizer.encode(full_text)

        # The target phrase tokens are the difference
        target_start_idx = len(prompt_tokens)
        target_tokens = full_tokens[target_start_idx:]

        # Extract logprobs for target tokens only
        # Note: prompt_logprobs_data is indexed from 1 (first token has no logprob)
        token_logprobs = []
        token_texts = []

        for i, token_id in enumerate(target_tokens):
            # The index in prompt_logprobs_data corresponds to position in full_tokens
            # Offset by 1 because first token has no logprob
            logprob_idx = target_start_idx + i

            if logprob_idx < len(prompt_logprobs_data) and prompt_logprobs_data[logprob_idx]:
                # Get the logprob for this specific token
                token_logprob_dict = prompt_logprobs_data[logprob_idx]
                if token_id in token_logprob_dict:
                    token_logprob = token_logprob_dict[token_id].logprob
                    token_logprobs.append(token_logprob)
                    token_texts.append(self.tokenizer.decode([token_id]))
                else:
                    # Token not in top-k logprobs, assign very low probability
                    token_logprobs.append(-100.0)
                    token_texts.append(self.tokenizer.decode([token_id]))
            else:
                # Fallback for missing data
                token_logprobs.append(-100.0)
                token_texts.append(self.tokenizer.decode([token_id]))

        # Calculate statistics
        if token_logprobs:
            import math
            total_logprob = sum(token_logprobs)
            avg_logprob = total_logprob / len(token_logprobs)
            # Approximate probability (may underflow for very low probabilities)
            try:
                probability = math.exp(total_logprob)
            except OverflowError:
                probability = 0.0

            # Perplexity = exp(-avg_logprob)
            try:
                perplexity = math.exp(-avg_logprob)
            except OverflowError:
                perplexity = float('inf')
        else:
            total_logprob = 0.0
            avg_logprob = 0.0
            probability = 0.0
            perplexity = float('inf')

        return {
            "total_logprob": total_logprob,
            "avg_logprob": avg_logprob,
            "probability": probability,
            "perplexity": perplexity,
            "num_tokens": len(token_logprobs),
            "tokens": target_tokens,
            "token_logprobs": token_logprobs,
            "token_texts": token_texts,
        }

    def cleanup(self):
        """Clean up vLLM resources."""
        if hasattr(self, 'llm'):
            # vLLM doesn't have explicit cleanup, but we can delete the object
            del self.llm
        if hasattr(self, 'tokenizer'):
            del self.tokenizer


class OpenAIModel(BaseModel):
    """OpenAI API model implementation (GPT-4, GPT-4o, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Initialize OpenAI model.

        Args:
            model_name: OpenAI model identifier (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: API key. If omitted, falls back to OPENROUTER_API_KEY when
                base_url points at OpenRouter, otherwise to OPENAI_API_KEY.
            base_url: Optional custom API base URL (OpenRouter, Azure, or any
                OpenAI-compatible endpoint). Defaults to OPENAI_BASE_URL if set.
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters passed to the OpenAI client
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is not installed. Install it with: pip install openai"
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            # OpenRouter and OpenAI use different keys; pick by endpoint.
            env_var = ("OPENROUTER_API_KEY"
                       if base_url and "openrouter" in base_url
                       else "OPENAI_API_KEY")
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(
                    f"No API key for {model_name!r}. Set {env_var} in your "
                    f"environment (see .env.example) or pass api_key explicitly."
                )

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        print(f"OpenAI model initialized: {model_name}")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response via the OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens)),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
        )
        if response.usage:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def generate_batch(self, prompt: str, num_samples: int = 3, **kwargs) -> list[str]:
        """Generate multiple responses via the OpenAI API using n parameter."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_new_tokens)),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            n=num_samples,
        )
        if response.usage:
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
        return [choice.message.content.strip() for choice in response.choices]

    def estimate_phrase_probability(
        self,
        prompt: str,
        target_phrase: str,
        use_chat_template: bool = True,
        num_samples: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate probability of a target phrase by sampling.

        Generates num_samples responses and counts how many start with
        the target phrase. The ratio approximates the phrase probability.

        Args:
            prompt: Input text prompt (context)
            target_phrase: The phrase whose probability we want to estimate
            use_chat_template: Ignored (OpenAI always uses chat format)
            num_samples: Number of samples to draw for estimation

        Returns:
            Dictionary with probability estimate (compatible with vLLM format)
        """
        import math

        responses = self.generate_batch(
            prompt, num_samples=num_samples,
            max_tokens=max(30, len(target_phrase.split()) * 4),
            temperature=kwargs.get("temperature", self.temperature),
        )

        target_lower = target_phrase.strip().lower()
        matches = sum(
            1 for r in responses if r.strip().lower().startswith(target_lower)
        )
        probability = matches / num_samples if num_samples > 0 else 0.0

        # Compute compatible fields
        total_logprob = math.log(probability) if probability > 0 else -20.0
        avg_logprob = total_logprob
        try:
            perplexity = math.exp(-avg_logprob)
        except OverflowError:
            perplexity = float('inf')

        return {
            "total_logprob": total_logprob,
            "avg_logprob": avg_logprob,
            "probability": probability,
            "perplexity": perplexity,
            "num_tokens": len(target_phrase.split()),
            "tokens": [],
            "token_logprobs": [],
            "token_texts": [],
            "num_samples": num_samples,
            "matches": matches,
        }

    def estimate_phrases_probability_batch(
        self,
        prompt: str,
        target_phrases: list[str],
        use_chat_template: bool = True,
        num_samples: int = 20,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Estimate probabilities for multiple phrases in a single batch of samples.

        Generates num_samples responses once, then checks each response against
        all target phrases. Much more efficient than calling estimate_phrase_probability
        per phrase.

        Args:
            prompt: Input text prompt (context)
            target_phrases: List of phrases whose probabilities we want to estimate
            use_chat_template: Ignored (OpenAI always uses chat format)
            num_samples: Number of samples to draw for estimation

        Returns:
            Dictionary mapping each phrase to its probability result dict
        """
        import math

        # Sample once, match against all phrases
        max_phrase_tokens = max(len(p.split()) for p in target_phrases) if target_phrases else 10
        responses = self.generate_batch(
            prompt, num_samples=num_samples,
            max_tokens=max(30, max_phrase_tokens * 4),
            temperature=kwargs.get("temperature", self.temperature),
        )

        responses_lower = [r.strip().lower() for r in responses]

        results = {}
        for phrase in target_phrases:
            target_lower = phrase.strip().lower()
            matches = sum(1 for r in responses_lower if r.startswith(target_lower))
            probability = matches / num_samples if num_samples > 0 else 0.0

            total_logprob = math.log(probability) if probability > 0 else -20.0
            avg_logprob = total_logprob
            try:
                perplexity = math.exp(-avg_logprob)
            except OverflowError:
                perplexity = float('inf')

            results[phrase] = {
                "total_logprob": total_logprob,
                "avg_logprob": avg_logprob,
                "probability": probability,
                "perplexity": perplexity,
                "num_tokens": len(phrase.split()),
                "tokens": [],
                "token_logprobs": [],
                "token_texts": [],
                "num_samples": num_samples,
                "matches": matches,
            }

        return results

    def cleanup(self):
        """No cleanup needed for API model."""
        pass


class MockModel(BaseModel):
    """
    Mock model for wiring checks without loading an LLM.

    The canned replies are shaped like real bargaining turns — a strategy tag,
    dialogue, and a priced Action — so a --use-mock run produces a
    multi-turn transcript with actual prices rather than terminating at turn
    one with no deal.

    It also implements the logprob interface, so mock runs exercise the same
    prior/acceptance code paths as a real vLLM run rather than silently falling
    back to static priors. The scores are a deterministic function of the phrase
    text: stable across runs, and non-uniform so that a uniform stub cannot mask
    a strategy-ordering bug. They carry no semantic meaning — a mock run tells
    you the pipeline is wired correctly and nothing more.
    """

    def __init__(self, responses: Optional[list] = None, accept_after: int = 4):
        """
        Initialize mock model.

        Args:
            responses: Cycle of canned replies. If given, prompt-awareness is
                disabled and the cycle is returned verbatim.
            accept_after: With the default replies, concede once the transcript
                embedded in the prompt has at least this many turns, so mock
                negotiations actually close instead of always hitting
                max_rounds.
        """
        self.responses = responses
        self.accept_after = accept_after
        self.call_count = 0
        self._default_cycle = [
            """[Propose a counter price] I'm interested, but that's above what I can pay.
Action: propose(price=$100)""",
            """That's below what I was hoping for on this one.
Action: counter(price=$150)""",
            """[Use comparatives] Comparable listings are going for noticeably less.
Action: propose(price=$120)""",
            """I could meet you part way here.
Action: counter(price=$135)""",
        ]

    def generate(self, prompt: str, **kwargs) -> str:
        """Return a mock response."""
        if self.responses is not None:
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return response

        # Concede once the conversation has run on a while. Cycling blindly
        # aliases badly: MCTS issues a fixed number of calls per turn, so a
        # pure round-robin lands on the same index every real turn and the
        # negotiation never closes.
        turns = prompt.count("Buyer:") + prompt.count("Seller:")
        if turns >= self.accept_after:
            self.call_count += 1
            return "That works for me — let's close at that price.\nAction: accept"

        response = self._default_cycle[self.call_count % len(self._default_cycle)]
        self.call_count += 1
        return response

    # -- logprob interface -------------------------------------------- #
    # MCTSNegotiator probes for these with hasattr. Without them, mock runs
    # take the static-prior and frequency-sampling fallbacks, which no real
    # run has ever used.

    @staticmethod
    def _score(phrase: str) -> float:
        """Deterministic pseudo-logprob in roughly [-3, -1], keyed on the text."""
        digest = sum((i + 1) * ord(c) for i, c in enumerate(phrase))
        return -1.0 - (digest % 200) / 100.0

    def estimate_phrase_probability(self, prompt: str, target_phrase: str, **kwargs) -> Dict[str, Any]:
        logprob = self._score(target_phrase)
        return {
            "total_logprob": logprob * max(len(target_phrase.split()), 1),
            "avg_logprob": logprob,
            "probability": math.exp(logprob),
            "perplexity": math.exp(-logprob),
            "tokens": target_phrase.split(),
            "token_logprobs": [logprob] * max(len(target_phrase.split()), 1),
            "token_texts": target_phrase.split(),
        }

    def estimate_phrases_probability_batch(
        self, prompt: str, target_phrases: list, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        return {p: self.estimate_phrase_probability(prompt, p) for p in target_phrases}

    def cleanup(self):
        """No cleanup needed for mock model."""
        pass
