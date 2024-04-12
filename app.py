import time
import gradio as gr
from llm import infer


def infer_with_llm(img, question):
    start_time = time.time()
    answer = infer(img, question)
    latency = time.time() - start_time

    return answer, latency


demo = gr.Interface(
    title="OmniLLM-12B",
    description="A playground for OmniLLM-12B",
    fn=infer_with_llm,
    inputs=[gr.Image(label="Image", type="filepath"), gr.Textbox(label="Question")],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Number(label="Latency(s)"),
    ],
    article="""[https://github.com/OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)""",
)

demo.launch(share=True)
