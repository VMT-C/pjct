import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
import re
import os
from dotenv import load_dotenv
import sympy # Import SymPy

# --- Configuration ---
# Load environment variables (for HF_TOKEN if needed)
load_dotenv()

# Model to use:
# Option 1 (Recommended if you have 16GB+ VRAM): CogBase-USTC/Qwen2.5-Math-7B-Instruct-SocraticLM
# Option 2 (If you have access and enough VRAM): CogBase-USTC/Llama-3.1-8B-Instruct-SocraticLM
# Option 3 (If you have less VRAM, try this with load_in_4bit=True): A smaller instruction-tuned model.
# We'll use Qwen2.5-Math-7B-Instruct-SocraticLM as the primary example.
FINE_TUNED_MODEL_IDENTIFIER = "CogBase-USTC/Qwen2.5-Math-7B-Instruct-SocraticLM"

# Set to True if the model requires trust_remote_code (Qwen often does)
TRUST_REMOTE_CODE = True

# Set to True to enable 4-bit quantization for lower VRAM usage.
# Requires bitsandbytes library and a compatible NVIDIA GPU.
# If set to False, model loads in float16/float32 (higher VRAM/RAM).
LOAD_IN_4BIT = True # Change to False if you have enough VRAM or are on CPU and hit issues with bitsandbytes

# If the model is gated (like Llama models often are), you'll need your Hugging Face token.
# Get it from https://huggingface.co/settings/tokens (ensure it has 'read' role).
# Then create a .env file in your project folder with: HF_TOKEN="hf_YOUR_TOKEN_HERE"
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN") # Will be None if not set in .env

# --- Initialize the Model ---
generator = None
tokenizer = None

print(f"Loading Socratic Math model: {FINE_TUNED_MODEL_IDENTIFIER}...")
try:
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        FINE_TUNED_MODEL_IDENTIFIER,
        token=HUGGING_FACE_TOKEN,
        trust_remote_code=TRUST_REMOTE_CODE
    )

    # Configure quantization if LOAD_IN_4BIT is True
    bnb_config = None
    if LOAD_IN_4BIT and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # Recommended for 4-bit
            bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16 if your GPU supports it
            bnb_4bit_use_double_quant=True,
        )
        print("Model will be loaded with 4-bit quantization (GPU only).")
    elif LOAD_IN_4BIT and not torch.cuda.is_available():
        print("Warning: LOAD_IN_4BIT is True, but no CUDA GPU detected. Model will fall back to CPU and may be slow.")
        LOAD_IN_4BIT = False # Disable 4-bit if no GPU

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        FINE_TUNED_MODEL_IDENTIFIER,
        quantization_config=bnb_config if LOAD_IN_4BIT else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() and not LOAD_IN_4BIT else torch.float32, # Use float16 if GPU and no 4-bit
        device_map="auto", # Automatically distributes model across available GPUs/CPU
        token=HUGGING_FACE_TOKEN,
        trust_remote_code=TRUST_REMOTE_CODE
    )

    # Create the text generation pipeline
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        # device is handled by device_map="auto", do not set it here explicitly
    )
    set_seed(42)
    print("Socratic Math model loaded successfully.")

except Exception as e:
    print(f"Error loading model {FINE_TUNED_MODEL_IDENTIFIER}: {e}")
    print("Possible reasons:")
    print("  - Incorrect model identifier or typo.")
    print("  - Not enough RAM/VRAM on your system to load the model (try LOAD_IN_4BIT=True if on GPU).")
    print("  - Missing 'token' if it's a gated model (check your .env and Hugging Face settings).")
    print("  - Missing `trust_remote_code=True` for this specific model type.")
    print("  - Network issues if downloading for the first time.")
    print("Exiting.")
    exit()

# --- NEW: Function to parse and solve math problems using SymPy ---
def parse_and_solve_equation(problem_string):
    """
    Attempts to parse and solve a simple algebraic equation from a string using SymPy.
    Returns the solution as a string (e.g., "x = 3") or None if parsing/solving fails.
    """
    try:
        # Define the symbol 'x'
        x = sympy.symbols('x')

        # Attempt to extract the equation part (e.g., "2x + 5 = 11")
        # This regex is a simple attempt; more robust parsing would be needed for complex inputs.
        match = re.search(r'([a-zA-Z0-9\s\+\-\*\/\=\(\)\.]+)', problem_string)
        if not match:
            return None # Could not find an equation-like string

        equation_str = match.group(0).replace(' ', '') # Remove spaces for easier parsing

        # Replace common math notations for SymPy
        equation_str = equation_str.replace('^', '**') # Power
        equation_str = equation_str.replace('=', '-') # Move everything to one side for Eq(expr, 0)

        # Create the SymPy expression. Assume it's an equation equal to 0.
        expr = sympy.sympify(equation_str)
        equation = sympy.Eq(expr, 0)

        # Solve the equation
        solutions = sympy.solve(equation, x)

        if solutions:
            # Format solutions nicely. Handle multiple solutions if they exist.
            if len(solutions) == 1:
                return f"x = {solutions[0]}"
            else:
                return f"x = {', '.join(str(sol) for sol in solutions)}"
        else:
            return None # No solutions found

    except (sympy.SympifyError, TypeError, ValueError, IndexError) as e:
        # Catch SymPy parsing errors or other issues
        print(f"SymPy parsing/solving error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in SymPy solver: {e}")
        return None

def generate_socratic_response(conversation_history, math_problem, current_student_answer=None):
    """
    Generates a Socratic question or feedback based on the conversation history and problem.
    """
    # The prompt is crucial. For fine-tuned models, try to match their training format.
    # This structure is common for instruction-tuned models.
    # Adjust this if the specific model's documentation suggests a different chat format.

    # Example for Qwen2.5-Math-7B-Instruct-SocraticLM, which usually follows a chat template.
    # The `apply_chat_template` method is the most robust way if available.
    messages = [
        {"role": "system", "content": "You are an expert AI math tutor who strictly uses the Socratic method. Your sole purpose is to guide the student to discover the solution to the problem themselves by asking thoughtful, probing questions. You must never give direct answers, solutions, or explicit steps. Every response must be a question or a statement leading to a question. If incorrect, help them identify the mistake. If correct, deepen understanding or lead to the next step. Be concise and encourage self-correction."},
        {"role": "user", "content": f"The math problem is: {math_problem}"}
    ]

    # Add previous conversation turns
    if conversation_history:
        # Simple parsing of history for this example.
        # In a real system, you'd want a more robust dialogue state manager.
        history_lines = conversation_history.strip().split('\n')
        for line in history_lines:
            if line.startswith("Student:"):
                messages.append({"role": "user", "content": line.replace("Student: ", "").strip()})
            elif line.startswith("Tutor:"):
                messages.append({"role": "assistant", "content": line.replace("Tutor: ", "").strip()})

    # Add the latest student response
    if current_student_answer:
        messages.append({"role": "user", "content": current_student_answer})

    # Use the model's specific chat template if available
    try:
        # `add_generation_prompt=True` tells the tokenizer to add tokens
        # that indicate the start of the assistant's response.
        socratic_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except AttributeError:
        # Fallback if tokenizer doesn't have apply_chat_template
        print("Warning: Tokenizer does not have 'apply_chat_template'. Using generic prompt.")
        socratic_prompt = f"""
    <|im_start|>system
    You are an expert AI math tutor who strictly uses the Socratic method. Your sole purpose is to guide the student to discover the solution to the problem themselves by asking thoughtful, probing questions. You must never give direct answers, solutions, or explicit steps. Every response must be a question or a statement leading to a question. If incorrect, help them identify the mistake. If correct, deepen understanding or lead to the next step. Be concise and encourage self-correction.<|im_end|>
    <|im_start|>user
    The math problem is: {math_problem}
    {conversation_history}
    Student's Latest Response: {current_student_answer if current_student_answer else "No specific response yet, just starting."}<|im_end|>
    <|im_start|>assistant
        """

    if not generator:
        return "Error: Model not loaded."

    try:
        response = generator(
            socratic_prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            truncation=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = response[0]['generated_text']

        # Extract only the model's response part after the prompt
        # This can be tricky depending on the model's output format.
        # Often, it's the text that comes *after* the input prompt.
        socratic_output = generated_text.replace(socratic_prompt, "").strip()

        # Basic post-processing to ensure it looks like a question or feedback
        if socratic_output and not (socratic_output.endswith('?') or socratic_output.endswith('.') or socratic_output.endswith('!')):
            socratic_output += '?'

        # If the model repeats the problem or student input, try to strip that
        socratic_output = socratic_output.split("Math Problem:")[0].strip()
        socratic_output = socratic_output.split("Current Conversation History:")[0].strip()
        socratic_output = socratic_output.split("Student's Latest Response:")[0].strip()

        return socratic_output

    except Exception as e:
        print(f"Error generating response: {e}")
        return "I am having trouble processing that. Can you rephrase?"


def run_chatbot():
    print("Welcome to the Socratic Math Tutor!")
    print("I will guide you through solving a math problem using questions.")
    print("Type 'quit' to exit.")

    math_problem = input("\nPlease enter the math problem you want to solve :\n")
    print(f"\nOkay, your problem for today: {math_problem}")

    # --- NEW: Dynamically solve the problem ---
    actual_solution = parse_and_solve_equation(math_problem)

    if actual_solution:
        print(f"Tutor (Internal Note: The solution is {actual_solution}).") # For debugging/info only
    else:
        print("\nTutor: I can't seem to parse or solve that problem internally right now. I'll still try to guide you Socratic-ally, but I won't be able to confirm the final answer.")
        print("Please try your question q.")
        # Set a placeholder so the loop can continue, but without a specific target
        actual_solution = "UNKNOWN"

    conversation_history = ""
    turns = 0
    max_turns = 15 # Give more turns for Socratic method

    # Start with an initial Socratic question
    initial_socratic_question = generate_socratic_response("", math_problem, "Let's begin.")
    print(f"Tutor: {initial_socratic_question}")
    conversation_history += f"Tutor: {initial_socratic_question}\n"


    while turns < max_turns:
        student_input = input("\nStudent: ")
        if student_input.lower() == 'quit':
            break

        conversation_history += f"Student: {student_input}\n"

        # --- UPDATED: Dynamic Final Answer Check ---
        # This part will now try to compare against the SymPy solution
        is_final_answer_attempt = "x=" in student_input.lower().replace(" ", "") or "the answer is" in student_input.lower()
        
        if is_final_answer_attempt and actual_solution != "UNKNOWN":
            # Normalize student input for comparison
            # This is a very basic normalization; a robust system would need more.
            normalized_student_input = student_input.lower().replace(" ", "").replace("theansweris", "").replace("x=", "")
            
            # Normalize SymPy solution for comparison (e.g., "x=3" -> "3")
            normalized_actual_solution = actual_solution.lower().replace(" ", "").replace("x=", "")
            
            # Check if student's answer matches the SymPy solution
            if normalized_student_input == normalized_actual_solution:
                print("Tutor: That's a great step! You've found the correct value for x. Can you explain the general steps you took to isolate x?")
                break # End the Socratic session on correct final answer
            else:
                print("Tutor: That's an interesting approach. It seems like we're not quite there yet. Can you walk me through your steps, specifically how you handled the numbers and variables to arrive at that result?")
        else:
            # Generate Socratic response using the LLM for intermediate steps
            socratic_response = generate_socratic_response(conversation_history, math_problem, student_input)
            print(f"Tutor: {socratic_response}")

        conversation_history += f"Tutor: {socratic_response}\n"
        turns += 1

    print("\nThanks for practicing math with me!")

if __name__ == "__main__":
    run_chatbot()
