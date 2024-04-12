import gradio as gr

def infer(img, question):
    print(img)

    return "1"


demo = gr.Interface(
    title="MiniCPM-V 2",
    description="A playground for MiniCPM-V",
    fn=infer,
    inputs=[gr.Image(label="Image"), gr.Textbox(label="Question")],
    outputs="text",
    article="""[https://github.com/OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)""",
)

demo.launch(share=True)
