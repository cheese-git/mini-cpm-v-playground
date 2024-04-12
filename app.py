import time
import gradio as gr


def infer(img, question):
    start_time = time.time()
    print(img)
    latency = time.time() - start_time

    return "1", latency


demo = gr.Interface(
    title="MiniCPM-V 2",
    description="A playground for MiniCPM-V",
    fn=infer,
    inputs=[gr.Image(label="Image"), gr.Textbox(label="Question")],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Number(label="Latency(s)"),
    ],
    article="""[https://github.com/OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)""",
)

demo.launch(share=True)
