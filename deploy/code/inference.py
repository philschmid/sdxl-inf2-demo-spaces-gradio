import torch
import torch_neuronx
import base64
from io import BytesIO
from optimum.neuron import NeuronStableDiffusionXLPipeline


def model_fn(model_dir):
    # load local converted model into pipeline
    pipeline = NeuronStableDiffusionXLPipeline.from_pretrained(model_dir)
    return pipeline


def predict_fn(data, pipeline):
    # extract prompt from data
    prompt = data.pop("inputs", data)
    prompts = [prompt] * 2 # two neuron core per worker


    parameters = data.pop("parameters", None)

    if parameters is not None:
        generated_images = pipeline(prompts, **parameters)["images"]
    else:
        generated_images = pipeline(prompts)["images"]

    # postprocess convert image into base64 string
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # always return the first
    return {"generated_images": encoded_images}
