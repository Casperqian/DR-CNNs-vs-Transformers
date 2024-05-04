import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from data.prepocess import GetTrainingData, GetTestData
from torch.utils.data import DataLoader
import argparse
import timm
from Algorithm.tta import T3A, TTFA
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import copy
import os

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.float().softmax(1) * x.float().log_softmax(1)).sum(1)

def generate_featurelized_loader(loader, featurizer, classifier, batch_size=1, device='cpu'):
        """
        The classifier adaptation does not need to repeat the heavy forward path,
        We speeded up the experiments by converting the observations into representations.
        """
        z_list = []
        y_list = []
        p_list = []
        featurizer.eval()
        classifier.eval()
        for x, y in loader:
            x = x.to(device)
            z = featurizer(x)
            p = classifier(z)

            z_list.append(z.detach().cpu())
            y_list.append(y.detach().cpu())
            p_list.append(p.detach().cpu())
        featurizer.train()
        classifier.train()
        z = torch.cat(z_list)
        y = torch.cat(y_list)
        p = torch.cat(p_list)
        ent = softmax_entropy(p)  # FloatTensor
        py = p.argmax(1).float().cpu().detach()
        dataset1, dataset2 = Dataset(z, y), Dataset(z, py)
        loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=True)
        loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=True)
        return loader1, loader2, ent

def adaptation(featurizer, classifier, loader, algorithm, adapt, device):
    predictions = []
    labels = []
    misclassifications = {'FP': 0, 'FN': 0}
    total_occurrences = {'control': 0, 'patient': 0}
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        featurizer.eval()
        classifier.eval()
        if adapt is None:
            p = algorithm(data)
        else:
            p = algorithm(data, adapt)
        torch.cuda.empty_cache()
        featurizer.train()
        classifier.train()
        _, predicted = torch.max(p.data, 1)
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

def inference(network, loader, device, step=2, beta=0.9, use_plc=False):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--root', type=str, default='E:\APTOS_RAW\colored_images')
    parser.add_argument('--network', type=str, default='efficientnet-b1',
                        help='networks: {resnet-34, resnet-50, efficientnet-b1, vit-t16, swinv2, convit-s}')
    parser.add_argument('--algorithm', type=str, default='TTFA')
    parser.add_argument('--use_cuda', type=bool, default=True)
    args = parser.parse_args()

    root = args.root
    if args.use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

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
    model = model.to(device)

    if args.network in ['vit-t16', 'vit-s16', 'swinv1', 'swinv2', 'convit-s']:
        classifier_name = 'head'
        classifier = copy.deepcopy(model.head.fc)
        model.head.fc = nn.Identity()
    elif args.network in ['efficientnet-b1', 'efficientnet-b5']:
        classifier_name = 'classifier'
        classifier = copy.deepcopy(model.classifier)
        model.classifier = nn.Identity()
    else:
        classifier_name = 'fc'
        classifier = copy.deepcopy(model.fc)
        model.fc = nn.Identity()
    featurizer = copy.deepcopy(model)
    featurizer = featurizer.to(device)
    classifier = classifier.to(device)
    n_outputs = classifier.in_features
    feature_loader, _, _ = generate_featurelized_loader(test_loader, featurizer, classifier, batch_size=32, device=device)

    # Vanilla
    # acc, auc, f1, mis_rate = inference(model, test_loader, device=device)
    # print('Vanilla Results:')
    # print('Test Accuracy: ', acc)
    # print('Test AUC: ', auc)
    # print('Test F1 score: ', f1)
    # print('Misclassification Rates for each class: ', mis_rate)

 ```# PLC
    # acc, auc, f1, mis_rate = inference(model, test_loader, device=device, step=2, use_plc=True)
    # print('Pseudo Label Calibration:')
    # print('Test Accuracy: ', acc)
    # print('Test AUC: ', auc)
    # print('Test F1 score: ', f1)
    # print('Misclassification Rates for each class: ', mis_rate)

    if args.algorithm == TTFA':
        tta = TTFA(featurizer, classifier, n_outputs, args)

    acc, auc, f1, mis_rate = adaptation(featurizer, classifier, feature_loader, algorithm=tta, adapt=True, device=device)
    print(args.algorithm)
    print('Test Accuracy: ', acc)
    print('Test AUC: ', auc)
    print('Test F1 score: ', f1)
    print('Misclassification Rates for each class: ', mis_rate)

   
