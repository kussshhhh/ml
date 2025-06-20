import os
import json
import random
import torch
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class TrainingDataGenerator:
    """Generate training data for function-calling model"""
    
    def __init__(self, executor: MathFunctionExecutor):
        self.executor = executor
    
    def generate_training_examples(self, num_examples: int = 1000) -> List[str]:
        """Generate training examples in the format we want"""
        examples = []
        
        # Basic arithmetic problems
        operations = [
            ('ADD', '+', self._generate_addition_pair),
            ('SUB', '-', self._generate_subtraction_pair),
            ('MUL', '*', self._generate_multiplication_pair),
            ('DIV', '/', self._generate_division_pair),
        ]
        
        for _ in range(num_examples):
            func_name, symbol, generator = random.choice(operations)
            a, b = generator()
            
            # Create training example - STOP at function call, don't include result
            # Format: "What is X + Y? The answer is ADD(X,Y) = "
            question_formats = [
                f"What is {a} {symbol} {b}?",
                f"Calculate {a} {symbol} {b}",
                f"Solve {a} {symbol} {b}",
                f"{a} {symbol} {b} equals",
                f"Find the result of {a} {symbol} {b}",
            ]
            
            question = random.choice(question_formats)
            answer = f" The answer is {func_name}({a},{b}) = "
            
            training_text = question + answer
            examples.append(training_text)
        
        # Add some single-number function examples (SQRT, ABS)
        for _ in range(num_examples // 5):
            if random.choice([True, False]):
                # SQRT
                num = random.randint(1, 100)
                # Use perfect squares sometimes
                if random.random() < 0.3:
                    sqrt_val = random.randint(1, 10)
                    num = sqrt_val ** 2
                
                question = random.choice([
                    f"What is the square root of {num}?",
                    f"Calculate sqrt({num})",
                    f"Find √{num}",
                ])
                answer = f" The answer is SQRT({num}) = "
            else:
                # ABS
                num = random.randint(-50, 50)
                
                question = random.choice([
                    f"What is the absolute value of {num}?",
                    f"Calculate |{num}|",
                    f"Find abs({num})",
                ])
                answer = f" The answer is ABS({num}) = "
            
            training_text = question + answer
            examples.append(training_text)
        
        return examples
    
    def _generate_addition_pair(self):
        return random.randint(1, 100), random.randint(1, 100)
    
    def _generate_subtraction_pair(self):
        a = random.randint(10, 100)
        b = random.randint(1, a)  # Ensure positive result
        return a, b
    
    def _generate_multiplication_pair(self):
        return random.randint(1, 20), random.randint(1, 20)
    
    def _generate_division_pair(self):
        # Generate clean divisions
        b = random.randint(2, 10)
        a = b * random.randint(1, 20)
        return a, b

class MathDataset(Dataset):
    """PyTorch dataset for math function calling training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def setup_model_and_tokenizer(model_name: str = "gpt2"):
    """Initialize model and tokenizer"""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def train_model(
    model,
    tokenizer,
    train_dataset,
    output_dir: str = "./function_calling_model",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    save_steps: int = 500,
    logging_steps: int = 100
):
    """Train the model"""
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        logging_steps=logging_steps,
        logging_dir=f"{output_dir}/logs",
        learning_rate=learning_rate,
        warmup_steps=100,
        report_to=None,  # Disable wandb/tensorboard logging
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed!")
    return trainer

def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Generate training data
    logger.info("Generating training data...")
    executor = MathFunctionExecutor()
    data_generator = TrainingDataGenerator(executor)
    
    # Generate training examples
    training_texts = data_generator.generate_training_examples(num_examples=2000)
    
    # Save training data for inspection
    with open("training_data.json", "w") as f:
        json.dump(training_texts[:10], f, indent=2)  # Save first 10 examples
    logger.info(f"Generated {len(training_texts)} training examples")
    logger.info("Sample training data saved to training_data.json")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer("gpt2")
    model.to(device)
    
    # Create dataset
    logger.info("Creating dataset...")
    train_dataset = MathDataset(training_texts, tokenizer, max_length=128)
    
    # Train the model
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir="./function_calling_model",
        num_epochs=3,
        batch_size=4,  # Reduced for smaller GPUs
        learning_rate=5e-5
    )
    
    # Test the trained model with a simple example
    logger.info("Testing trained model...")
    model.eval()
    
    test_prompt = "What is 15 + 7?"
    inputs = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 20,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Test input: {test_prompt}")
    logger.info(f"Generated: {generated_text}")
    
    logger.info("Training and testing completed successfully!")

if __name__ == "__main__":
    main()