import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_completions_and_rewards(transformer_model, reward_model, num_trials=10):
    # Load the tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model)
    generator_model = AutoModelForCausalLM.from_pretrained(transformer_model)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model)

    # Load the IMDB dataset
    dataset = load_dataset("imdb", split="train")
    prompts = [x["text"] for x in dataset.select(range(10))]

    # Initialize pipelines
    generator = pipeline("text-generation", model=generator_model, tokenizer=tokenizer)
    results = []
    # Process each prompt to generate text and evaluate it
    for prompt in tqdm(prompts):
        # Tokenizing and truncating to first 64 tokens
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=64
        )["input_ids"]
        prompt_text = tokenizer.decode(input_ids[0])

        # Generating completions
        set_seed(42)  # For reproducibility
        completions = generator(
            prompt_text,
            max_length=64 + 48,
            min_length=64,
            num_return_sequences=2,
            top_k=50,
            do_sample=True,
        )
        rewards = []
        completion_texts = []
        # Calculate rewards for the completions
        print(f"Prompt: {prompt_text}")
        for i, completion in enumerate(completions):
            generated_text = completion["generated_text"][
                len(prompt_text) :
            ]  # Only the generated part
            rew_inputs = reward_tokenizer(
                generated_text, return_tensors="pt", truncation=True, max_length=512
            )
            # Get the logits from the reward model
            with torch.no_grad():
                outputs = reward_model(**rew_inputs)
                logits = outputs.logits
                # Assuming positive sentiment is the second class
                reward = logits[0][1].item()
            rewards.append(reward)
            completion_texts.append(generated_text)

        # Compute softmax probabilities
        probabilities = softmax(np.array(rewards))

        # Generate Bernoulli samples
        bernoulli_samples = np.random.binomial(
            1, probabilities, (num_trials, len(probabilities))
        )

        # Prepare the result for this set of completions
        result = {
            "prompt": prompt_text,
            "completion1": completion_texts[0],
            "completion2": completion_texts[1],
            "reward1": rewards[0],
            "reward2": rewards[1],
        }
        for i in range(num_trials):
            # Updated keys and values interpretation for "winning" completions
            result[f"winning_completion_{i+1}"] = bernoulli_samples[
                i, 1
            ]  # Reflects which completion won

        results.append(result)

    return results


# Example usage
transformer_model = "lvwerra/gpt2-imdb"  # Replace with your chosen transformer model
reward_model = "lvwerra/distilbert-imdb"  # Replace with your reward model
dataset = generate_completions_and_rewards(transformer_model, reward_model)
for data in dataset:
    print(data)
    print("\n")
    input()
