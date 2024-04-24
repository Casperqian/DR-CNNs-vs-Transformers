import torch
import torchvision
from torch import nn
import os
import torch.nn.functional as F
from data.prepocess import GetTrainingData, GetTestData
from torch.utils.data import DataLoader
import argparse
import timm
import logging
from torch.utils.tensorboard import SummaryWriter
from data.utils import softmax_entropy
import ssl
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--root', type=str, default='E:\APTOS_RAW\colored_images')
    parser.add_argument('--network', type=str, default='vit-t16',
                        help='networks: {resnet-34, resnet-50, efficientnet-b1, vit-t16, swinv2, convit-s}')
    args = parser.parse_args()

    root = args.root

    test_dataset = GetTestData(root)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print('Test batches:', len(test_loader))

    if args.network == 'resnet-34':
        model = torchvision.models.resnet34()
        model.fc = nn.Linear(512, 5)
        ckpt = torch.load(r'./results/DR66k_resnet-34_20_acc0.8468_auc0.8846.pth')
    elif args.network == 'resnet-50':
        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, 5)
        ckpt = torch.load(r'./results/DR66k_resnet-50_20_acc0.8573_auc0.8936.pth')
    elif args.network == 'efficientnet-b1':
        model = torchvision.models.efficientnet_b1()
        model.classifier = nn.Linear(1280, 5)
        ckpt = torch.load(r'.\results\DR66k_efficientnet-b1_20_acc0.8724_auc0.9067.pth')
    elif args.network == 'vit-t16':
        model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=False)
        model.reset_classifier(num_classes=5)
        ckpt = torch.load(r'.\results\DR_66k_vit-t16_20_acc0.8412_auc0.8815.pth')
    elif args.network == 'vit-s16':
        model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=False)
        model.reset_classifier(num_classes=5)
        ckpt = torch.load(r'.\results\DR66k_vit-s16_20_acc0.8387_auc0.8823.pth')
    elif args.network == 'swinv2':
        model = timm.create_model('swinv2_cr_tiny_ns_224.sw_in1k', pretrained=False)
        model.reset_classifier(num_classes=5)
        ckpt = torch.load(r'./results/DR66k_swinv2_best_acc0.8515_auc0.8872.pth')
    elif args.network == 'convit-s':
        model = timm.create_model('convit_small.fb_in1k', pretrained=False)
        model.reset_classifier(num_classes=5)
        ckpt = torch.load(r'./results/DK66k_convit-s_15_acc0.8447_auc0.8842.pth')

    model.load_state_dict(ckpt)
    model = model.to('cuda')


    def collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def inference(network, loader, device, step=2, beta=0.9, use_plc=False, use_label=False):
        # params, names = collect_params(network)
        optimizer = torch.optim.SGD(network.parameters(), lr=1e-4, weight_decay=0.0001)
        predictions = []
        labels = []
        misclassifications = {'FP': 0, 'FN': 0}
        total_occurrences = {'control': 0, 'patient': 0}
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            network.train()
            if use_plc:
                for _ in range(step):
                    optimizer.zero_grad()
                    outputs = network(data)
                    # loss = 0.1 * softmax_entropy(outputs).mean(0)
                    py, y_prime = F.softmax(outputs, dim=-1).max(1)
                    flag = (py > beta)
                    loss = F.cross_entropy(outputs[flag], y_prime[flag])
                    loss.backward()
                    optimizer.step()
            elif use_label:
                for _ in range(step):
                    optimizer.zero_grad()
                    outputs = network(data)
                    py, y_prime = F.softmax(outputs, dim=-1).max(1)
                    flag = (py < beta)
                    loss = F.cross_entropy(outputs[flag], target[flag])
                    loss.backward()
                    optimizer.step()
            network.eval()
            with torch.no_grad():
                outputs = network(data)
            _, predicted = torch.max(outputs.data, 1)
            for i, (pred, label) in enumerate(zip(predicted, target)):
                if (label.item() in [1, 2, 3] and pred.item() in [0, 4]) or \
                        (label.item() == 4 and pred.item() in [0, 1, 2, 3]):
                    misclassifications['FN'] += 1
                if label.item() == 0 and pred.item() != 0:
                    misclassifications['FP'] += 1

                if label.item() == 0:
                    total_occurrences['control'] += 1
                else:
                    total_occurrences['patient'] += 1
            predictions.extend(predicted.tolist())
            labels.extend(target.tolist())

        misclassification_rates = {
            'FPR': misclassifications['FP'] / total_occurrences['control'],
            'FNR': misclassifications['FN'] / total_occurrences['patient'],
        }

        labels = F.one_hot(torch.tensor(labels), num_classes=5).numpy()
        predictions = F.one_hot(torch.tensor(predictions), num_classes=5).numpy()
        accuracy = accuracy_score(labels, predictions)
        auc = roc_auc_score(labels, predictions, multi_class='ovo')
        f1 = f1_score(labels, predictions, average='macro')

        return accuracy, auc, f1, misclassification_rates


    # Vanilla
    acc, auc, f1, mis_rate = inference(model, test_loader, device='cuda')
    print('Vanilla Results:')
    print('Test Accuracy: ', acc)
    print('Test AUC: ', auc)
    print('Test F1 score: ', f1)
    print('Misclassification Rates for each class: ', mis_rate)

    # PLC
    acc, auc, f1, mis_rate = inference(model, test_loader, device='cuda', step=2, use_plc=True)

    # Threshold Label
    # acc, auc, f1, mis_rate = inference(model, test_loader, device='cuda', step=2, use_label=True)
    print('Pseudo Label Calibration:')
    print('Test Accuracy: ', acc)
    print('Test AUC: ', auc)
    print('Test F1 score: ', f1)
    print('Misclassification Rates for each class: ', mis_rate)
