import os
from dotenv import load_dotenv

load_dotenv()
# If running on AutoDL, set HF_HOME to /autodl-fs
if os.environ["AutoDLContainerUUID"]:
    os.environ["HF_HOME"] = "/root/autodl-fs/.cache/huggingface"

import json
import torch
from chat import OmniLMMChat, img2base64

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.float16 if device == "mps" else torch.bfloat16
torch.manual_seed(0)

print("Initializing model...")
chat_model = OmniLMMChat("openbmb/MiniCPM-V-2")
print("Model initialized.")

def infer(img, question):
    """
    Args:
        img: Image file path

    """
    img_64 = img2base64(img)
    msgs = [{"role": "user", "content": question}]
    inputs = {"image": img_64, "question": json.dumps(msgs)}

    print("Start infering...")
    answer = chat_model.chat(inputs)

    return answer


if __name__ == "__main__":

    answer = infer(img="./assets/hk_OCR.jpg", question="有几辆车")
    print(answer)
