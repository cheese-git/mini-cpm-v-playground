import json
from dotenv import load_dotenv

load_dotenv()

import torch
from chat import OmniLMMChat, img2base64

torch.manual_seed(0)

model = OmniLMMChat("openbmb/OmniLMM-12B")
# model = model.to(device="mps", dtype=torch.float16)


def infer(img, question):
    msgs = [{"role": "user", "content": question}]

    answer = model.chat({"image": img, "question": json.dumps(msgs)})

    return answer


if __name__ == "__main__":
    img = img2base64("./assets/hk_OCR.jpg")

    print(img)

    # answer = infer(

    # )
