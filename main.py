import streamlit as st
from uuid import uuid4
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BKvLLM:
    """
    GPT-2 Model Class for OpenAI Community GPT-2 using Hugging Face Transformers

    Class for GPT-2 inference using Hugging Face's `transformers` library.
    """

    def __init__(self,
                 model: str,
                 max_model_len: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.2,
                 max_tokens: int = 256,
                 min_tokens: int = 128):

        # Load the tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.model = GPT2LMHeadModel.from_pretrained(model)

        self.model.eval()  # Set the model to evaluation mode

        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def generate(self, prompt: str) -> str:
        """
        Generate Text using the GPT-2 model.

        Parameters
        ----------
        `prompt` : `str` : raw input text

        Return
        ------
        `str` : generated text from the model
        """

        # Tokenize the input prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate text using the model
        output = self.model.generate(
            inputs,
            max_length=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )

        # Decode and return the generated text
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# Streamlit app
st.title("OpenAI GPT-2 Text Generation")
st.write("This is an example of text generation using OpenAI's GPT-2 model.")

# User input for prompt
prompt = st.text_area("Enter prompt:", "Apa itu kecerdasan buatan?")

# User input for temperature
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

# Generate button
if st.button("Generate Text"):
    # Initialize the model
    llm = BKvLLM(model="openai-community/gpt2", temperature=temperature)
    
    # Generate output text
    output = llm.generate(prompt)
    
    # Display the generated text
    st.subheader("Generated Text")
    st.write(output)
