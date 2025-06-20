import re
import torch
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionCall:
    name: str
    args: List[float]
    start_pos: int
    end_pos: int
    result: float

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
    
    def add(self, args: List[float]) -> float:
        return sum(args)
    
    def subtract(self, args: List[float]) -> float:
        if len(args) < 2:
            raise ValueError("SUB requires at least 2 arguments")
        result = args[0]
        for arg in args[1:]:
            result -= arg
        return result
    
    def multiply(self, args: List[float]) -> float:
        result = 1
        for arg in args:
            result *= arg
        return result
    
    def divide(self, args: List[float]) -> float:
        if len(args) != 2:
            raise ValueError("DIV requires exactly 2 arguments")
        if args[1] == 0:
            raise ValueError("Division by zero")
        return args[0] / args[1]
    
    def power(self, args: List[float]) -> float:
        if len(args) != 2:
            raise ValueError("POW requires exactly 2 arguments")
        return args[0] ** args[1]
    
    def sqrt(self, args: List[float]) -> float:
        if len(args) != 1:
            raise ValueError("SQRT requires exactly 1 argument")
        if args[0] < 0:
            raise ValueError("Cannot take square root of negative number")
        return args[0] ** 0.5
    
    def absolute(self, args: List[float]) -> float:
        if len(args) != 1:
            raise ValueError("ABS requires exactly 1 argument")
        return abs(args[0])
    
    def execute(self, func_name: str, args: List[float]) -> float:
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        return self.functions[func_name](args)

class FunctionCallDetector:
    """Detects and parses function calls in text"""
    
    def __init__(self, executor: MathFunctionExecutor):
        self.executor = executor
        # Pattern to match function calls like ADD(1,2,3) optionally followed by " = "
        self.pattern = r'([A-Z]+)\(([^)]+)\)(\s*=\s*)?'
    
    def parse_args(self, args_str: str) -> List[float]:
        """Parse comma-separated arguments"""
        try:
            args = [float(arg.strip()) for arg in args_str.split(',')]
            return args
        except ValueError as e:
            raise ValueError(f"Invalid arguments: {args_str}")
    
    def detect_incomplete_functions(self, text: str) -> List[Dict]:
        """Find function calls that need results appended"""
        functions = []
        
        for match in re.finditer(self.pattern, text):
            func_name = match.group(1)
            args_str = match.group(2)
            equals_part = match.group(3)  # " = " if present
            
            try:
                args = self.parse_args(args_str)
                result = self.executor.execute(func_name, args)
                
                # Check if this function call already has a result
                has_equals = equals_part is not None
                
                functions.append({
                    'name': func_name,
                    'args': args,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'result': result,
                    'has_equals': has_equals,
                    'full_match': match.group(0)
                })
            except ValueError as e:
                logger.warning(f"Invalid function call: {func_name}({args_str}) - {e}")
                continue
        
        return functions
    
    def append_results_to_functions(self, text: str) -> Tuple[str, List[Dict]]:
        """Append calculation results to function calls"""
        functions = self.detect_incomplete_functions(text)
        
        # Filter to only functions that need results appended
        incomplete_functions = []
        for func in functions:
            if func['has_equals']:
                # Check if there's already a number after the equals
                end_pos = func['end_pos']
                remaining_text = text[end_pos:].strip()
                
                # If there's no number immediately after "= ", this function needs completion
                if not remaining_text or not remaining_text[0].isdigit():
                    incomplete_functions.append(func)
            else:
                # No equals sign, needs " = result" appended
                incomplete_functions.append(func)
        
        if not incomplete_functions:
            return text, []
        
        # Sort by position (reverse order to maintain positions)
        incomplete_functions.sort(key=lambda f: f['start_pos'], reverse=True)
        
        modified_text = text
        for func in incomplete_functions:
            result_str = str(int(func['result'])) if func['result'].is_integer() else f"{func['result']:.2f}"
            
            if func['has_equals']:
                # Append result after existing " = "
                modified_text = modified_text[:func['end_pos']] + result_str + modified_text[func['end_pos']:]
            else:
                # Add " = result" after the function call
                modified_text = modified_text[:func['end_pos']] + f" = {result_str}" + modified_text[func['end_pos']:]
        
        return modified_text, incomplete_functions

class FunctionCallingInference:
    """Main inference class for function-calling model"""
    
    def __init__(self, model_path: str = "./function_calling_model"):
        """Initialize the inference system"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize function executor and detector
        self.executor = MathFunctionExecutor()
        self.detector = FunctionCallDetector(self.executor)
        
        logger.info("Model loaded successfully!")
    
    def generate_with_functions(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Generate text with function calling support.
        This implements the core idea: generate -> detect functions -> execute -> continue
        """
        
        generation_log = []
        current_text = prompt
        total_generated = ""
        
        for iteration in range(max_iterations):
            logger.info(f"Generation iteration {iteration + 1}")
            
            # Tokenize current text
            inputs = self.tokenizer.encode(current_text, return_tensors='pt').to(self.device)
            
            # Generate continuation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,  # Generate 30 more tokens
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    early_stopping=True
                )
            
            # Decode the generated text
            full_generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_part = full_generated[len(current_text):]
            
            generation_log.append({
                'iteration': iteration + 1,
                'input': current_text,
                'generated_part': new_part,
                'full_text': full_generated
            })
            
            # Check for function calls in the newly generated part that need completion
            processed_part, function_calls = self.detector.append_results_to_functions(new_part)
            
            if function_calls:
                logger.info(f"Found {len(function_calls)} function calls")
                for func in function_calls:
                    logger.info(f"  {func['name']}({func['args']}) = {func['result']}")
                
                # Update the current text with processed functions
                current_text = full_generated.replace(new_part, processed_part)
                total_generated += processed_part
                
                # Continue generation if we found functions
                continue
            else:
                # No function calls found, we're done
                total_generated += new_part
                break
        
        # Final result
        final_text = prompt + total_generated
        
        return {
            'prompt': prompt,
            'final_output': final_text,
            'generated_part': total_generated,
            'iterations': len(generation_log),
            'generation_log': generation_log,
        }
    
    def simple_generate(self, prompt: str, max_length: int = 100) -> str:
        """Simple generation without function calling (for comparison)"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def test_model(self):
        """Test the model with various prompts"""
        test_prompts = [
            "What is 15 + 7?",
            "Calculate 12 * 8",
            "What is 100 - 25?",
            "Find 144 / 12",
            "What is the square root of 64?",
            "Calculate the absolute value of -15",
            "Solve 5 + 3 * 2",  # This should be interesting
        ]
        
        logger.info("Testing model with various prompts...")
        
        for prompt in test_prompts:
            print(f"\n{'='*50}")
            print(f"PROMPT: {prompt}")
            print('='*50)
            
            # Test with function calling
            result = self.generate_with_functions(prompt)
            print(f"WITH FUNCTIONS: {result['final_output']}")
            
            # Test without function calling (raw model)
            simple_result = self.simple_generate(prompt, max_length=50)
            print(f"WITHOUT FUNCTIONS: {simple_result}")
            
            if result['generation_log']:
                print(f"ITERATIONS: {result['iterations']}")
                for log_entry in result['generation_log']:
                    if 'generated_part' in log_entry:
                        print(f"  Iter {log_entry['iteration']}: {log_entry['generated_part']}")

def interactive_mode():
    """Interactive mode for testing the model"""
    print("=== Interactive Function-Calling Model ===")
    print("Loading model...")
    
    try:
        inference = FunctionCallingInference()
        print("Model loaded successfully!")
        print("\nEnter math questions or type 'quit' to exit")
        print("Examples: 'What is 5 + 3?', 'Calculate 12 * 7', etc.")
        print("-" * 50)
        
        while True:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print("Model: Generating response...")
            result = inference.generate_with_functions(prompt)
            
            print(f"Model: {result['final_output']}")
            
            # Show function calls if any were made
            if result['iterations'] > 1:
                print(f"(Used {result['iterations']} generation steps with function calls)")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've trained the model first using train_function_calling_model.py")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        # Run tests
        try:
            inference = FunctionCallingInference()
            inference.test_model()
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure you've trained the model first using train_function_calling_model.py")
            print("Usage: python inference_function_calling.py [interactive]")

if __name__ == "__main__":
    main()
