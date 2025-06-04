from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class Generator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def generate_answer(self, query, context_chunks, max_length=150):
        prompt = "Context:\n" + "\n".join(context_chunks) + "\n\nQuestion:\n" + query + "\nAnswer:"
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        outputs = self.model.generate(inputs, max_length=max_length, 
                                      num_return_sequences=1, 
                                      no_repeat_ngram_size=2,
                                      pad_token_id=self.tokenizer.eos_token_id)

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt part from answer
        answer = answer[len(prompt):].strip()
        return answer
