import json
from dotenv import load_dotenv

load_dotenv()

import torch
from chat import OmniLMMChat, img2base64

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.bfloat16
torch.manual_seed(0)

model = OmniLMMChat("openbmb/OmniLMM-12B")
model = model.to(device, dtype=dtype)

def infer(img, question):
    """
    Args:
        img: Image file path
    
    """
    img_64 = img2base64(img)
    msgs = [{"role": "user", "content": question}]

    answer = model.chat({"image": img_64, "question": json.dumps(msgs)})

    return answer


if __name__ == "__main__":

    answer = infer(
      img="./assets/hk_OCR.jpg",
      question="有几辆车"
    )
