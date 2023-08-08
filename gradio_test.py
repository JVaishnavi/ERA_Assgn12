import torch, torchvision
from torchvision import transforms
import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from lt_model_new import LT_model
model = LT_model()
model.load_state_dict(torch.load("model_state_dict_new.pt", map_location = torch.device("cpu")), strict = False)

model = torch.load("lt_model_new.pth")

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def inference(input_img, grad_cam = "Yes", opacity = 0.5, layer = -1, no_classes = 3):
    transform = transforms.ToTensor()
    input_img = input_img.resize((32,32))
    org_img = input_img
    input_img = transform(input_img)
    input_img = input_img.unsqueeze(0)
    outputs = model(input_img)
    softmax = torch.nn.Softmax(dim=0)
    o = softmax(outputs.flatten())
    confidence = {classes[i]: float(o[i]) for i in range(10)}
    confidence = dict(sorted(confidence.items(), key = lambda x: x[1], reverse=True)[:no_classes])
    if grad_cam == "Yes":
        target_layers = [model.layer_3] if layer==-1 else [model.layer_2]
        cam = GradCAM(model = model, target_layers=target_layers, use_cuda = False)
        targets = [ClassifierOutputTarget(torch.argmax(outputs))]
        grayscale_cam = cam(input_tensor = input_img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img = input_img.squeeze(0)
        rgb_img = np.transpose(img, [1,2,0]).numpy()
        visualisation = show_cam_on_image(org_img/255, grayscale_cam, use_rgb = True, image_weight = opacity)
    else:
        visualisation = org_img
        
    return confidence, visualisation
    
demo = gr.Interface(
    inference,
    inputs = [
        
        gr.Image(shape = (32, 32), label = "Input Image"),
        gr.Dropdown(["Yes", "No"], value = "Yes", label = "Do you want gradCAM images?"),
        gr.Slider(0, 1, value = 0.5, label = "Overlay Opacity of Image"),
        gr.Slider(-2, -1, value = -1, step = 1, label = "Which layer for gradCAM?"),
        gr.Slider(0, 10, value = 3, step = 1, label = "How many top classes do you want to see?"),
    ],

    outputs = [
        gr.Label(),
        gr.Image(shape = (32, 32), label = "Output").style(width = 128, height = 128)
    ],
    
    title = "CIFAR10 data trained on custom model with gradCAM",
    description = "Session 12 ERA assignment",
    examples = [
        ["images/cat1.jpeg", "Yes", 0.5, -1, 5],
        ["images/dog1.jpeg", "No", 1, -2, 9],
        ["images/bird1.jpg", "Yes", 0.5, -2, 4],
        ['images/car1.jpeg', 'Yes', 0.8, -1, 7],
        ['images/plane1.jpg', 'Yes', 0.8, -1, 7],
        ['images/deer1.jpg', 'Yes', 0.8, -1, 6],
        ['images/frog1.jpg', 'Yes', 0.8, -1, 7],
        ['images/truck1.jpg', 'Yes', 0.8, -1, 8],
        ['images/ship1.jpeg', 'Yes', 0.8, -1, 7],
        ['images/horse1.jpg', 'Yes', 0.8, -1, 7]
    ]
)

demo.launch()