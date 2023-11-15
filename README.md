# bbang-hyung-end
# 고해상도 이미지 생성 Stable Diffusion 
코드 실행 본 : https://colab.research.google.com/drive/1eb9h1PN7HM7oxQKwFMxxUddM5XG_Q1zt#scrollTo=IGAMauVyjFU-

```python
!nvidia-smi
```
## 파이썬 패키지 설치
5분정도 소요
```python
!pip install --upgrade -qq git+https://github.com/huggingface/diffusers.git transformers accelerate scipy xformers gradio translate
```

![image](https://github.com/hsy0511/bbang-hyung-end/assets/104752580/91bb0212-9873-434f-a157-90a77ee11353)

## 모델 다운로드 및 로드
```python
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, height=768, width=768).images[0]
    
image.save("astronaut_rides_horse.png")

image
```

![image](https://github.com/hsy0511/bbang-hyung-end/assets/104752580/a2cc93e1-09a8-4e86-9c0c-758ef43fb1e6)

![image](https://github.com/hsy0511/bbang-hyung-end/assets/104752580/e4e86d43-2b3e-4fbb-831c-70303c4f43b3)

## 영어 번역 및 이미지 생성 함수 정의
```python
from translate import Translator

translator = Translator(from_lang="ko", to_lang="en")

def generate_images(text):
    print(text)
    translation = translator.translate(text)
    print(translation)

    image = pipe(translation, height=768, width=768).images[0]

    return image

generate_images("펭귄이 곰한테 뽀뽀하는 모습")
```

![image](https://github.com/hsy0511/bbang-hyung-end/assets/104752580/9d27afd4-7c69-4ada-bde2-17ec4b266286)

## input output으로 이미지 생성하기
```python
import gradio as gr

def inference(text):
    return generate_images(text)

demo = gr.Interface(fn=inference, inputs="text", outputs="image")

demo.launch(debug=True, share=True)
```

![image](https://github.com/hsy0511/bbang-hyung-end/assets/104752580/bcc7ef56-58dc-4b05-a302-c562de7a7b96)

![image](https://github.com/hsy0511/bbang-hyung-end/assets/104752580/5c6b326e-9911-4c65-92fe-c7ff7dfa9e89)
