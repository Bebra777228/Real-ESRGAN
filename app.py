import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model2 = RealESRGAN(device, scale=2)
model2.load_weights('weights/RealESRGAN_x2.pth', download=True)
if torch.cuda.device_count() > 1:
    model2 = torch.nn.DataParallel(model2)

model4 = RealESRGAN(device, scale=4)
model4.load_weights('weights/RealESRGAN_x4.pth', download=True)
if torch.cuda.device_count() > 1:
    model4 = torch.nn.DataParallel(model4)

model8 = RealESRGAN(device, scale=8)
model8.load_weights('weights/RealESRGAN_x8.pth', download=True)
if torch.cuda.device_count() > 1:
    model8 = torch.nn.DataParallel(model8)

def inference(image, size):
    global model2
    global model4
    global model8
    if image is None:
        raise gr.Error("Image not uploaded")

    width, height = image.size
    if width >= 5000 or height >= 5000:
        raise gr.Error("The image is too large.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if size == '2x':
        try:
            result = model2.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model2 = RealESRGAN(device, scale=2)
            model2.load_weights('weights/RealESRGAN_x2.pth', download=False)
            result = model2.predict(image.convert('RGB'))
    elif size == '4x':
        try:
            result = model4.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model4 = RealESRGAN(device, scale=4)
            model4.load_weights('weights/RealESRGAN_x4.pth', download=False)
            result = model4.predict(image.convert('RGB'))
    else:
        try:
            result = model8.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model8 = RealESRGAN(device, scale=8)
            model8.load_weights('weights/RealESRGAN_x8.pth', download=False)
            result = model8.predict(image.convert('RGB'))

    print(f"Image size ({device}): {size} ... OK")
    return result

title = "Real ESRGAN UpScale"

gr.Interface(inference,
    [gr.Image(type="pil"),
    gr.Radio(['2x', '4x', '8x'],
    type="value",
    value='2x',
    label='Resolution model')],
    gr.Image(type="pil", label="Output"),
    title= title,
    allow_flagging='never',
    cache_examples=False,
).queue(api_open=True).launch(show_error=True, show_api=True, share=True)
