import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
import re
import os
from dotenv import load_dotenv
import sympy
import math
import streamlit as st

# --- Configuration ---
# Load environment variables (for HF_TOKEN if needed)
load_dotenv()

# Model to use:
FINE_TUNED_MODEL_IDENTIFIER = "CogBase-USTC/Qwen2.5-Math-7B-Instruct-SocraticLM"

# Set to True if the model requires trust_remote_code (Qwen often does)
TRUST_REMOTE_CODE = True

# Set to True to enable 4-bit quantization for lower VRAM usage.
LOAD_IN_4BIT = True

# Get Hugging Face token from environment variables
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")

# --- Cached Model Initialization ---
@st.cache_resource
def load_socratic_model():
    """Loads the Socratic Math LLM and tokenizer."""
    global LOAD_IN_4BIT

    st.info(f"Loading Socratic Math model: {FINE_TUNED_MODEL_IDENTIFIER}... This might take a few minutes.")
    try:
        tokenizer_obj = AutoTokenizer.from_pretrained(
            FINE_TUNED_MODEL_IDENTIFIER,
            token=HUGGING_FACE_TOKEN,
            trust_remote_code=TRUST_REMOTE_CODE
        )

        bnb_config = None
        if LOAD_IN_4BIT and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            st.info("Model will be loaded with 4-bit quantization (GPU only).")
        elif LOAD_IN_4BIT and not torch.cuda.is_available():
            st.warning("LOAD_IN_4BIT is True, but no CUDA GPU detected. Model will fall back to CPU and may be slow.")
            LOAD_IN_4BIT = False

        model_obj = AutoModelForCausalLM.from_pretrained(
            FINE_TUNED_MODEL_IDENTIFIER,
            quantization_config=bnb_config if LOAD_IN_4BIT else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() and not LOAD_IN_4BIT else torch.float32,
            device_map="auto",
            token=HUGGING_FACE_TOKEN,
            trust_remote_code=TRUST_REMOTE_CODE
        )

        generator_obj = pipeline(
            'text-generation',
            model=model_obj,
            tokenizer=tokenizer_obj,
        )
        set_seed(42)
        st.success("Socratic Math model loaded successfully!")
        return generator_obj, tokenizer_obj

    except Exception as e:
        st.error(f"Error loading model {FINE_TUNED_MODEL_IDENTIFIER}: {e}")
        st.error("Possible reasons:")
        st.error(" - Incorrect model identifier or typo.")
        st.error(" - Not enough RAM/VRAM on your system to load the model (try LOAD_IN_4BIT=True if on GPU).")
        st.error(" - Missing 'token' if it's a gated model (check your .env and Hugging Face settings).")
        st.error(" - Missing `trust_remote_code=True` for this specific model type.")
        st.error(" - Network issues if downloading for the first time.")
        st.stop()

# --- Algebraic Solver Function ---
def parse_and_solve_algebraic(problem_string):
    """
    Attempts to parse and solve a simple algebraic equation from a string using SymPy.
    More robustly handles non-algebraic input by returning None early.
    """
    # Pre-check: If the string doesn't contain common math operators/digits, it's likely a word problem
    # that SymPy can't directly parse. This prevents SymPy from throwing an internal error.
    if not re.search(r'[\d=+\-*/xya-z]', problem_string.lower()):
        return None # Not likely an algebraic expression

    try:
        equations_str = [eq.strip() for eq in problem_string.split(',') if eq.strip()]
        if not equations_str:
            return None

        sym_equations = []
        all_variables = set()

        for eq_str in equations_str:
            # The existing regex `re.search(r'([a-zA-Z0-9\s\+\-\*\/\=\(\)\.]+)', eq_str)`
            # is quite broad. Let's ensure `sympy.sympify` receives clean input.
            
            # Additional check: ensure the string primarily consists of algebraic characters
            if not re.fullmatch(r'[\w\s+\-*/=().^]+', eq_str):
                return None # Contains characters that are not part of an equation

            cleaned_eq_str = eq_str.replace(' ', '')
            cleaned_eq_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', cleaned_eq_str)
            cleaned_eq_str = re.sub(r'(\d)(\()', r'\1*\2', cleaned_eq_str)
            cleaned_eq_str = re.sub(r'(\))(\()', r'\1*\2', cleaned_eq_str)
            cleaned_eq_str = re.sub(r'([a-zA-Z])(\()', r'\1*\2', cleaned_eq_str)
            cleaned_eq_str = cleaned_eq_str.replace('^', '**')

            if '=' in cleaned_eq_str:
                lhs, rhs = cleaned_eq_str.split('=')
                expr = sympy.sympify(f"({lhs}) - ({rhs})")
            else:
                expr = sympy.sympify(cleaned_eq_str)

            sym_equations.append(expr)
            all_variables.update(expr.free_symbols)

        if not all_variables:
            return None

        variables_to_solve = sorted(list(all_variables), key=str)
        solutions = sympy.solve(sym_equations, variables_to_solve)

        if solutions:
            if isinstance(solutions, dict):
                return ", ".join(f"{str(k)} = {sympy.N(v, 5)}" for k, v in solutions.items())
            elif isinstance(solutions, list):
                if not solutions:
                    return None
                
                formatted_solutions = []
                for sol_set in solutions:
                    if isinstance(sol_set, dict):
                        formatted_solutions.append(", ".join(f"{str(k)} = {sympy.N(v, 5)}" for k, v in sol_set.items()))
                    elif isinstance(sol_set, (tuple, list)):
                        formatted_sol_set = []
                        for i, val in enumerate(sol_set):
                            if i < len(variables_to_solve):
                                formatted_sol_set.append(f"{variables_to_solve[i]} = {sympy.N(val, 5)}")
                        formatted_solutions.append(", ".join(formatted_sol_set))
                    else:
                        if len(variables_to_solve) == 1:
                            formatted_solutions.append(f"{variables_to_solve[0]} = {sympy.N(sol_set, 5)}")

                return "; ".join(formatted_solutions)
            else:
                return str(sympy.N(solutions, 5))
        else:
            return None

    except (sympy.SympifyError, TypeError, ValueError, IndexError) as e:
        # print(f"SymPy algebraic parsing/solving error: {e}") # You can uncomment for debugging
        return None # Return None on specific parsing/solving errors
    except Exception as e:
        # print(f"An unexpected error occurred in SymPy algebraic solver: {e}") # You can uncomment for debugging
        return None # Catch any other unexpected errors and return None

# --- Geometric Solver Function ---
def parse_and_solve_geometry(problem_string):
    """
    Attempts to parse and solve basic geometric problems (area, perimeter, circumference) for common shapes.
    """
    problem_lower = problem_string.lower()
    
    # Extract numbers from the string
    numbers = [float(n) for n in re.findall(r'\d+\.?\d*', problem_lower)]

    # Rectangle / Square
    if "rectangle" in problem_lower or "square" in problem_lower:
        length, width, side = None, None, None

        len_match = re.search(r'(length|l)\s*(\d+\.?\d*)', problem_lower)
        wid_match = re.search(r'(width|w)\s*(\d+\.?\d*)', problem_lower)
        side_match = re.search(r'(side|s)\s*(\d+\.?\d*)', problem_lower)

        if len_match: length = float(len_match.group(2))
        if wid_match: width = float(wid_match.group(2))
        if side_match: side = float(side_match.group(2))

        if "square" in problem_lower:
            if side is not None:
                length = side
                width = side
            elif len(numbers) >= 1:
                length = numbers[0]
                width = numbers[0]
        
        elif length is None and width is None and len(numbers) >= 2:
            length, width = numbers[0], numbers[1]

        if length is not None and width is not None:
            if "area" in problem_lower:
                return f"Area = {length * width}"
            elif "perimeter" in problem_lower:
                return f"Perimeter = {2 * (length + width)}"

    # Circle
    elif "circle" in problem_lower:
        radius = None
        rad_match = re.search(r'(radius|r)\s*(\d+\.?\d*)', problem_lower)
        if rad_match:
            radius = float(rad_match.group(2))
        elif len(numbers) >= 1:
            radius = numbers[0]

        if radius is not None:
            if "area" in problem_lower:
                return f"Area = {math.pi * (radius ** 2):.2f}"
            elif "circumference" in problem_lower or "perimeter" in problem_lower:
                return f"Circumference = {2 * math.pi * radius:.2f}"

    # Triangle (Area only for now)
    elif "triangle" in problem_lower and "area" in problem_lower:
        base, height = None, None
        
        base_match = re.search(r'(base|b)\s*(\d+\.?\d*)', problem_lower)
        height_match = re.search(r'(height|h)\s*(\d+\.?\d*)', problem_lower)

        if base_match: base = float(base_match.group(2))
        if height_match: height = float(height_match.group(2))

        if base is not None and height is not None:
            return f"Area = {0.5 * base * height}"
        elif len(numbers) >= 2:
            return f"Area = {0.5 * numbers[0] * numbers[1]}"

    return None

# --- Universal Problem Solver Dispatcher ---
def parse_and_solve_universal(problem_string):
    """
    Attempts to solve the problem, first as a geometric problem, then as an algebraic one.
    """
    # Try geometric first
    solution = parse_and_solve_geometry(problem_string)
    if solution:
        return solution
    
    # If not geometric, try algebraic
    solution = parse_and_solve_algebraic(problem_string)
    return solution

# --- Generate Socratic Response ---
def generate_socratic_response(conversation_history, math_problem, current_student_answer=None):
    """
    Generates a Socratic question or feedback based on the conversation history and problem.
    """
    generator = st.session_state.generator
    tokenizer = st.session_state.tokenizer

    messages = [
        {"role": "system", "content": (
            "You are an expert AI math tutor specializing in the Socratic method. "
            "Your goal is to patiently guide the student to discover solutions through clear, direct, and non-judgmental questions. "
            "Never provide direct answers, formulas, or explicit step-by-step solutions unless explicitly told to do so (i.e., after multiple incorrect attempts). "
            "Every response must be a question or a statement leading to a question. "
            "Break down concepts into the smallest possible logical steps. "
            "Focus on one question or idea per turn. "
            "If the student is incorrect, gently re-direct their thinking by asking them to re-examine their assumptions or a specific part of their work, without explicitly stating 'wrong'. "
            "If the student provides a correct step or answer, acknowledge their progress positively and then ask 'how' or 'why' questions to deepen their understanding of the process. "
            "Maintain a calm and encouraging tone."
        )},
        {"role": "user", "content": f"The math problem is: {math_problem}"}
    ]

    # Add previous conversation turns
    if conversation_history:
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
        socratic_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except AttributeError:
        st.warning("Tokenizer does not have 'apply_chat_template'. Using generic prompt.")
        socratic_prompt = f"""
<|im_start|>system
You are an expert AI math tutor who strictly uses the Socratic method. Your sole purpose is to guide the student to discover the solution to the problem themselves by asking thoughtful, probing questions. You must never give direct answers, solutions, or explicit steps unless explicitly told to do so (i.e., after multiple incorrect attempts). Every response must be a question or a statement leading to a question. If incorrect, help them identify the mistake. If correct, deepen understanding or lead to the next step. Be concise and encourage self-correction.<|im_end|>
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
        socratic_output = generated_text.replace(socratic_prompt, "").strip()

        # Basic post-processing
        if socratic_output and not (socratic_output.endswith('?') or socratic_output.endswith('.') or socratic_output.endswith('!')):
            socratic_output += '?'

        # Strip repeated context
        socratic_output = socratic_output.split("Math Problem:")[0].strip()
        socratic_output = socratic_output.split("Current Conversation History:")[0].strip()
        socratic_output = socratic_output.split("Student's Latest Response:")[0].strip()

        return socratic_output

    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I am having trouble processing that. Can you rephrase?"

# --- Streamlit App Logic ---
def main():
    st.set_page_config(page_title="Socratic Math Tutor", layout="centered")
    st.title("Socratic Math Tutor 🎓")
    st.write("I will guide you through solving a math problem using questions. Type 'quit' to exit the session at any time.")

    # --- Accessibility Settings Sidebar ---
    st.sidebar.header("Accessibility Settings")
    
    # Font Size
    font_size = st.sidebar.slider("Adjust Font Size", 12, 24, 16)
    st.markdown(f"<style>body {{ font-size: {font_size}px; }} .st-chat-message p {{ font-size: {font_size}px; }}</style>", unsafe_allow_html=True)

    # Color Theme
    theme_options = {
        "Light": {"bg": "#FFFFFF", "text": "#333333", "primary": "#4CAF50", "secondary": "#E8F5E9"},
        "Dark": {"bg": "#1E1E1E", "text": "#FFFFFF", "primary": "#66BB6A", "secondary": "#333333"},
        "High Contrast": {"bg": "#000000", "text": "#FFFF00", "primary": "#00FF00", "secondary": "#333333"},
        "Calm Blue": {"bg": "#E0F2F7", "text": "#212121", "primary": "#2196F3", "secondary": "#BBDEFB"}
    }
    selected_theme_name = st.sidebar.selectbox("Select Theme", list(theme_options.keys()))
    selected_theme = theme_options[selected_theme_name]
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {selected_theme["bg"]};
            color: {selected_theme["text"]};
        }}
        .stButton>button {{
            background-color: {selected_theme["primary"]};
            color: {selected_theme["text"]};
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background-color: {selected_theme["secondary"]};
            color: {selected_theme["primary"]};
        }}
        .stTextInput>div>div>input {{
            background-color: {selected_theme["secondary"]};
            color: {selected_theme["text"]};
            border-radius: 8px;
            padding: 10px;
        }}
        .st-chat-message-user {{
            background-color: {selected_theme["secondary"]};
            border-radius: 10px;
            padding: 10px;
        }}
        .st-chat-message-assistant {{
            background-color: #F0F0F0;
            border-radius: 10px;
            padding: 10px;
        }}
        .st-chat-message-assistant p {{
            color: black !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # --- JavaScript for Text-to-Speech (always enabled without a checkbox) ---
    tts_js = """
    <script>
    function speak(text) {
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }
        if (window.speechSynthesis) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
        }
    }
    </script>
    """
    st.markdown(tts_js, unsafe_allow_html=True)

    # Initialize session state variables
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "math_problem" not in st.session_state:
        st.session_state.math_problem = ""
    if "actual_solution" not in st.session_state:
        st.session_state.actual_solution = "UNKNOWN"
    if "conversation_history_raw" not in st.session_state:
        st.session_state.conversation_history_raw = ""
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "incorrect_attempts" not in st.session_state:
        st.session_state.incorrect_attempts = 0
    if "problem_solved" not in st.session_state: # New flag
        st.session_state.problem_solved = False


    # Load model (only runs once due to @st.cache_resource)
    if not st.session_state.model_loaded:
        with st.spinner("Loading Socratic Math model..."):
            st.session_state.generator, st.session_state.tokenizer = load_socratic_model()
            st.session_state.model_loaded = True
            st.rerun()

    # --- Problem Input Section ---
    if not st.session_state.math_problem:
        st.subheader("Enter Your Math Problem")
        problem_input = st.text_input(
            "Please type the **initial math problem** you want to solve here "
            "(e.g., 'Solve for x: 3x - 7 = 8', 'Area of rectangle length 5 width 3'):",
            key="problem_text_input"
        )
        
        if st.button("Start Tutoring Session", key="start_button") and problem_input:
            st.session_state.math_problem = problem_input
            st.session_state.actual_solution = parse_and_solve_universal(problem_input)
            st.session_state.incorrect_attempts = 0 # Reset attempts for new problem
            st.session_state.problem_solved = False # Reset solved flag

            if st.session_state.actual_solution is None:
                st.warning("Tutor: I can't seem to parse or solve that problem internally right now. I'll still try to guide you Socratic-ally, but I won't be able to confirm the final answer.")
                st.session_state.actual_solution = "UNKNOWN"
            else:
                st.success(f"Tutor (Internal Note: The solution is {st.session_state.actual_solution}).")

            # Start the conversation with the first Socratic question
            initial_socratic_question = generate_socratic_response(
                st.session_state.conversation_history_raw,
                st.session_state.math_problem,
                "Let's begin."
            )
            st.session_state.messages.append({"role": "assistant", "content": initial_socratic_question})
            st.session_state.conversation_history_raw += f"Tutor: {initial_socratic_question}\n"
            
            # Text-to-speech for initial question (always on)
            st.markdown(f"<script>speak('{initial_socratic_question.replace("'", "\\'")}')</script>", unsafe_allow_html=True)
            
            st.rerun()

    # --- Chat Interface Section ---
    if st.session_state.math_problem:
        st.subheader(f"Problem: {st.session_state.math_problem}")

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Prevent further input if problem is solved
        if st.session_state.problem_solved:
            st.info("You've found the solution! To work on a new problem, click 'Start a New Problem'.")
            if st.button("Start a New Problem", key="new_problem_button_after_solve"):
                st.session_state.math_problem = ""
                st.session_state.messages = []
                st.session_state.conversation_history_raw = ""
                st.session_state.actual_solution = "UNKNOWN"
                st.session_state.incorrect_attempts = 0
                st.session_state.problem_solved = False
                st.rerun()
            return # Stop execution here to prevent input box from showing

        # Chat input for student
        student_input = st.chat_input("Type your answer or next step here (type 'quit' to start a new problem):", key="student_chat_input")

        if student_input:
            st.session_state.messages.append({"role": "user", "content": student_input})
            st.session_state.conversation_history_raw += f"Student: {student_input}\n"

            # Check for 'quit' command
            if student_input.lower() == 'quit':
                goodbye_message = "Thanks for practicing math with me! Goodbye."
                with st.chat_message("assistant"):
                    st.markdown(goodbye_message)
                # Text-to-speech for goodbye message (always on)
                st.markdown(f"<script>speak('{goodbye_message.replace("'", "\\'")}')</script>", unsafe_allow_html=True)
                
                # Reset session state
                st.session_state.math_problem = ""
                st.session_state.messages = []
                st.session_state.conversation_history_raw = ""
                st.session_state.actual_solution = "UNKNOWN"
                st.session_state.incorrect_attempts = 0
                st.session_state.problem_solved = False
                st.rerun()

            # --- Dynamic Final Answer Check ---
            tutor_response = ""
            if st.session_state.actual_solution != "UNKNOWN":
                # Normalize actual solution and student input for comparison
                normalized_actual = st.session_state.actual_solution.lower().strip().replace(" ", "")
                normalized_student = student_input.lower().strip().replace(" ", "")
                
                is_correct = False

                # Try to extract a numerical value from the actual solution for comparison
                actual_numerical_value = None
                # Matches numbers at the end, handling "Area=15.0" -> 15.0
                num_match_actual = re.search(r'([\d\.\-]+)$', normalized_actual) 
                if num_match_actual:
                    try:
                        actual_numerical_value = float(num_match_actual.group(1))
                    except ValueError:
                        pass

                # Check 1: Direct comparison of full string (e.g., "area=36", "x=5")
                if normalized_actual == normalized_student:
                    is_correct = True
                # Check 2: If student just provides the numerical answer and it matches the extracted numerical value
                elif actual_numerical_value is not None:
                    try:
                        student_numerical_input = float(normalized_student)
                        if abs(actual_numerical_value - student_numerical_input) < 1e-5: # Tolerance for floats
                            is_correct = True
                    except ValueError:
                        pass # Student input is not a simple number

                if is_correct:
                    tutor_response = "Good job! You have found the solution. Let's go for the next question."
                    st.session_state.incorrect_attempts = 0 # Reset counter on correct answer
                    st.session_state.problem_solved = True # Mark problem as solved
                else:
                    st.session_state.incorrect_attempts += 1
                    if st.session_state.incorrect_attempts >= 4: # If 4 or more incorrect attempts (0-indexed: 0, 1, 2, 3)
                        tutor_response = f"It looks like you're having some difficulty with this problem. Sometimes a fresh perspective helps. The correct answer is: **{st.session_state.actual_solution}**. Would you like to start a new problem, or would you like to review the concepts related to this problem?"
                        st.session_state.incorrect_attempts = 0 # Reset for future interactions or new problem
                        st.session_state.problem_solved = True # Mark as solved after revealing answer
                    else:
                        tutor_response = generate_socratic_response(
                            st.session_state.conversation_history_raw,
                            st.session_state.math_problem,
                            student_input
                        )
            else:
                # If actual_solution is UNKNOWN (SymPy/Geometric solver failed), always defer to LLM
                tutor_response = generate_socratic_response(
                    st.session_state.conversation_history_raw,
                    st.session_state.math_problem,
                    student_input
                )
            
            st.session_state.messages.append({"role": "assistant", "content": tutor_response})
            st.session_state.conversation_history_raw += f"Tutor: {tutor_response}\n"
            
            # Text-to-speech for tutor response (always on)
            st.markdown(f"<script>speak('{tutor_response.replace("'", "\\'")}')</script>", unsafe_allow_html=True)
            
            st.rerun()

if __name__ == "__main__":
    main()