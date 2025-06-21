import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class MathFunctionExecutor:
    """Handles execution of mathematical functions"""
    
    def __init__(self):
        self.functions = {
            'ADD': self.add,
            'SUB': self.subtract,
            'MUL': self.multiply,
            'DIV': self.divide,
            'POW': self.power,
            'SQRT': self.sqrt,
            'ABS': self.absolute
        }
    
    def add(self, args): return sum(args)
    def subtract(self, args): return args[0] - sum(args[1:]) if len(args) >= 2 else args[0]
    def multiply(self, args): 
        result = 1
        for arg in args: result *= arg
        return result
    def divide(self, args): return args[0] / args[1] if len(args) == 2 and args[1] != 0 else 0
    def power(self, args): return args[0] ** args[1] if len(args) == 2 else 0
    def sqrt(self, args): return args[0] ** 0.5 if len(args) == 1 and args[0] >= 0 else 0
    def absolute(self, args): return abs(args[0]) if len(args) == 1 else 0
    
    def execute(self, func_name, args):
        if func_name in self.functions:
            return self.functions[func_name](args)
        return 0

class SimpleMathModel:
    def __init__(self, model_path="./function_calling_model"):
        """Load the trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.executor = MathFunctionExecutor()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, prompt, max_length=50):
        """Generate response and execute any function calls"""
        
        # Generate text
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                early_stopping=True
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract and execute function calls
        response = self._process_functions(generated_text)
        
        return response
    
    def _process_functions(self, text):
        """Find function calls and replace with results"""
        
        # Pattern to match function calls like ADD(15,7)
        pattern = r'([A-Z]+)\(([^)]+)\)'
        
        def replace_function(match):
            func_name = match.group(1)
            args_str = match.group(2)
            
            try:
                # Parse arguments
                args = [float(x.strip()) for x in args_str.split(',')]
                # Execute function
                result = self.executor.execute(func_name, args)
                return str(int(result) if result == int(result) else result)
            except:
                return match.group(0)  # Return original if parsing fails
        
        # Replace all function calls with their results
        processed_text = re.sub(pattern, replace_function, text)
        
        # Clean up the response - remove everything after "="
        if " = " in processed_text:
            # Find the pattern "FUNCTION(args) = " and replace with just the result
            pattern2 = r'([A-Z]+\([^)]+\))\s*=\s*(\d+(?:\.\d+)?)'
            processed_text = re.sub(pattern2, r'\2', processed_text)
            
            # If there's still garbage after the number, clean it
            parts = processed_text.split(" = ")
            if len(parts) > 1:
                # Take everything before " = " and add the clean result
                before_equals = parts[0]
                # Extract just the number result
                result_match = re.search(r'\b(\d+(?:\.\d+)?)\b', parts[1])
                if result_match:
                    return before_equals + " " + result_match.group(1)
        
        return processed_text

def main():
    """Simple chat interface"""
    print("Loading model...")
    model = SimpleMathModel()
    print("Model loaded! Type 'quit' to exit.\n")
    
    while True:
        prompt = input("you: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if prompt:
            response = model.generate_response(prompt)
            print(f"gpt2: {response}\n")

if __name__ == "__main__":
    main()