import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(scale):
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model

def deep_clean_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def inference(image, size):
    if image is None:
        raise gr.Error("Image not uploaded")

    width, height = image.size
    if width >= 5000 or height >= 5000:
        raise gr.Error("The image is too large.")

    try:
        if size == '2x':
            model = load_model(2)
        elif size == '4x':
            model = load_model(4)
        else:
            model = load_model(8)

        result = model.predict(image.convert('RGB'))
        print(f"Image size ({device}): {size} ... OK")
        return result
    except torch.cuda.OutOfMemoryError as e:
        print(e)
        raise gr.Error("Out of GPU memory. Please try with a smaller image or lower resolution model.")
    finally:
        deep_clean_gpu()

title = "Real ESRGAN UpScale"

gr.Interface(inference,
    [gr.Image(type="pil"),
    gr.Radio(['2x', '4x', '8x'],
    type="value",
    value='2x',
    label='Resolution model')],
    gr.Image(type="pil", label="Output"),
    title=title,
    css="footer{display:none !important}",
    allow_flagging='never',
    cache_examples=False,
).launch(share=True)
