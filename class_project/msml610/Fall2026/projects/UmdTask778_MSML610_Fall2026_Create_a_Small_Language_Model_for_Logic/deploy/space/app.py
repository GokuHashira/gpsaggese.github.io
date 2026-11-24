"""
Gradio demo: small from-scratch language models solving math word problems and
logic questions step by step (MSML610 class project, UMD, Fall 2026).
"""

import gradio as gr

import solver

MODELS = solver.models()
MATH_EXAMPLES = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
    "in May. How many clips did Natalia sell altogether in April and May?",
    "A train travels 60 miles per hour for 3 hours and then 40 miles per hour for 2 hours. "
    "How far does it travel in total?",
    "12 * (7 + 5) - 36 / 4 = ?",
]
LOGIC_EXAMPLE = (
    "Erin is smart. All smart people are rough. If Erin is rough then Erin is big. "
    "Big, smart people are blue.",
    "Erin is blue",
)


def run_math(model_name, problem, sample):
    out = solver.solve(MODELS[model_name], solver.math_prompt(problem), sample=sample)
    return out["steps"], out["answer"], f"{out['tokens']} tokens in {out['seconds']} s"


def run_logic(model_name, premises, question, sample):
    out = solver.solve(MODELS[model_name], solver.logic_prompt(premises, question), sample=sample)
    return out["steps"], out["answer"], f"{out['tokens']} tokens in {out['seconds']} s"


with gr.Blocks(title="Small LMs for Math and Logic") as demo:
    gr.Markdown(
        "# Small Language Models for Math and Logic\n"
        "Models trained **from scratch** (119M–1.31B parameters) on open math data, then "
        "fine-tuned. They write step-by-step solutions ending in a final answer. "
        "Runs on CPU, so larger models take longer."
    )
    model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Model")
    sample = gr.Checkbox(False, label="Sample (temperature 0.7) instead of greedy")
    with gr.Tab("Math"):
        problem = gr.Textbox(lines=4, label="Problem")
        gr.Examples(MATH_EXAMPLES, inputs=problem)
        btn = gr.Button("Solve", variant="primary")
        steps = gr.Textbox(lines=10, label="Worked solution")
        answer = gr.Textbox(label="Final answer")
        speed = gr.Markdown()
        btn.click(run_math, [model, problem, sample], [steps, answer, speed])
    with gr.Tab("Logic"):
        premises = gr.Textbox(lines=4, label="Facts and rules", value=LOGIC_EXAMPLE[0])
        question = gr.Textbox(label="Statement to check", value=LOGIC_EXAMPLE[1])
        btn2 = gr.Button("Check", variant="primary")
        steps2 = gr.Textbox(lines=10, label="Proof")
        answer2 = gr.Textbox(label="Answer (yes / no / unknown)")
        speed2 = gr.Markdown()
        btn2.click(run_logic, [model, premises, question, sample], [steps2, answer2, speed2])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
