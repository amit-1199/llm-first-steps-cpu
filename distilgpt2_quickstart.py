# This script shows how to download a text-generation model from Hugging Face
# and run prompts using only CPU (no CUDA/GPU required).
#
# Why this approach?
# Many institutes and tutorials provide code that requires CUDA (GPU) and expect you to use Google Colab.
# However, not everyone explains how to run these models on your own computer using only CPU.
# This script is for those who want to experiment locally, without needing a GPU or Colab.
#
# Steps:
# 1. Choose a model from Hugging Face (here we use "distilgpt2" for speed and low resource usage).
# 2. Download the tokenizer and model using AutoTokenizer and AutoModelForCausalLM.
# 3. Create a text-generation pipeline.
# 4. Run your prompt and print the generated output.
#
# You can change MODEL_ID to any supported Hugging Face model.
# For larger models, ensure you have enough RAM and compute resources.
# For interactive use, uncomment the loop at the end.

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

# Configure basic logging to display information during execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model and tokenizer (these can be initialized once)
# Choose a model from Hugging Face (here we use "distilgpt2" for speed and low resource usage)
MODEL_ID = "distilgpt2"
logger.info(f"Loading tokenizer for model: {MODEL_ID}")
# Download and load the tokenizer associated with the chosen model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
logger.info("Tokenizer loaded successfully.")

logger.info(f"Loading model: {MODEL_ID}")
# Download and load the pre-trained language model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
logger.info("Model loaded successfully.")

# Create text-generation pipeline
# This simplifies the process of text generation (tokenization, model inference, decoding)
logger.info("Creating text-generation pipeline.")
pipe = pipeline(
    "text-generation",
    model=model,  # Connect the loaded model
    tokenizer=tokenizer,  # Connect the loaded tokenizer
    device=-1  # Use CPU (-1 indicates CPU, 0 or higher would typically indicate a GPU)
)
logger.info("Pipeline created successfully.")

# Add a check to ensure the pipeline is callable
if not callable(pipe):
    logger.error("Text generation pipeline is not callable. There might be an issue with initialization.")
    # You might want to add more error handling or exit here if the pipeline is not usable

def run_prompt(prompt):
    # Print the prompt being processed
    print(f"Running inference for prompt: {prompt}")
    if callable(pipe):
        # Run the text generation using the configured pipeline
        out = pipe(
            prompt,
            max_new_tokens=250,  # Limit the maximum number of new tokens to generate
            do_sample=True,  # Enable sampling-based text generation for varied outputs
            temperature=0.7,  # Control the randomness of sampling (lower = less random, higher = more random)
            top_p=0.9,  # Use nucleus sampling (consider tokens whose cumulative probability exceeds 0.9)
            pad_token_id=tokenizer.eos_token_id, # Specify the token ID to use for padding
            no_repeat_ngram_size=2, # Prevent repeating n-grams of this size
        )
        print("Output:")
        # Extract and print the generated text from the output
        print(out[0]["generated_text"])
    else:
        print("Pipeline is not callable. Cannot run inference.")


if __name__ == "__main__":
    # Example usage
    prompt = "Give me three practical study tips for learning Python fast."
    run_prompt(prompt)


    # Uncomment below for interactive prompt loop
    # while True:
    #     prompt = input("Enter your prompt (or 'exit' to quit): ")
    #     if prompt.lower() == "exit":
    #         break
    #     run_prompt(prompt)
