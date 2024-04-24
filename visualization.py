import timm
import argparse
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import os
import glob
from data.utils import RGBA2RGB
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import cv2
import numpy as np
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='E:\APTOS_RAW\colored_images\\No_DR\\',
                        help='Input image path')
    parser.add_argument('--network', type=str, default='resnet-50',
                        help='network: {resnet-50}')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


if __name__ == "__main__":
    args = get_args()
    device = args.use_cuda

    if args.network == 'resnet-34':
        ckpt = torch.load(r'./results/DR66k_resnet-34_20_acc0.8468_auc0.8846.pth')
        model = torchvision.models.resnet34()
        model.fc = nn.Linear(512, 5)
    elif args.network == 'resnet-50':
        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, 5)
        ckpt = torch.load(r'./results/DR66k_resnet-50_20_acc0.8573_auc0.8936.pth')
    elif args.network == 'vit-s16':
        model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=False)
        model.reset_classifier(num_classes=5)
        ckpt = torch.load(r'.\results\DR66k_vit-s16_20_acc0.8387_auc0.8823.pth')
    elif args.network == 'swinv2':
        model = timm.create_model('swinv2_cr_tiny_ns_224.sw_in1k', pretrained=False)
        model.reset_classifier(num_classes=5)
        ckpt = torch.load(r'./results/DR66k_swinv2_best_acc0.8515_auc0.8872.pth')

    print(model)
    print('Loading model...')
    model.load_state_dict(ckpt)

    target_layer = [model.layer4[-1]]
    # target_layer = [model.norm]
    if args.use_cuda:
        model = model.to('cuda')

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    '''
    Resnet18 and 50: model.layer4[-1]
    VGG and densenet161: model.features[-1]
    mnasnet1_0: model.layers[-1]
    ViT: model.blocks[-1].norm1
    '''

    from scipy.ndimage import zoom

    img_dir = glob.glob(os.path.join(args.image_path, '*'))

    transform = transforms.Compose([
        RGBA2RGB(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for image_path in img_dir:
        class_name = image_path.split('\\')[-2]
        id = image_path.split('\\')[-1]

        print('class_name:', class_name)
        print('id:', id)
        class_path = os.path.join('E:/GradCAM', f'{args.network}', f'{class_name}')
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        img = Image.open(image_path)
        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.to('cuda')
        model.eval()
        logits = model(input_tensor).softmax(-1).cpu().detach().numpy()
        model.train()
        logits = np.round(logits, 2)

        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  
        rgb_img = cv2.imread(image_path, 1)
        rgb_img = zoom(rgb_img, (0.5, 0.5, 1))
        rgb_img = np.float32(rgb_img) / 255

        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])  # torch.Size([1, 3, 224, 224])

        cam = GradCAM(model=model, target_layers=target_layer)
        grayscale_cam = cam(input_tensor=input_tensor)  # [batch, 224,224]

        grayscale_cam = grayscale_cam.squeeze()
        visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)

        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        plt.imshow(visualization)
        # plt.show()
        plt.savefig(os.path.join(class_path, f'{id}_logits{logits}.png'), bbox_inches='tight', pad_inches=0)
