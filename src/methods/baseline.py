import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class BaselineModel:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct", use_4bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Quantification 4-bit 
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if not use_4bit else None,
            trust_remote_code=True
        )

    def generate_sql(self, system_msg: str, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.01,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._extract_sql(response)

    def _extract_sql(self, text: str) -> str:
        """Nettoie la sortie pour extraire uniquement le bloc SQL."""
        match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback si pas de balises markdown
        match_block = re.search(r"```(.*?)```", text, re.DOTALL)
        if match_block:
            return match_block.group(1).strip()
        return text.strip()
