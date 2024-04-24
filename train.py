import torch
import torchvision
from torch import nn
import os
import math
import torch.nn.functional as F
from data.prepocess import GetTrainingData, GetTestData
from torch.utils.data import DataLoader
import argparse
import timm
import logging
from torch.utils.tensorboard import SummaryWriter
import ssl
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
# from networks.mamba import Vim
from thop import profile
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    print('Version of PyTorch:', torch.__version__) # PyTorch: 1.10.0 + cuda 11.3
    parser = argparse.ArgumentParser(description='DR Classification')
    parser.add_argument('--root', type=str, default=r'E:\DR2015\colored_images\colored_images')
    parser.add_argument('--dataset', type=str, default='DR2015')
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--base_lr', type=int, default=1e-3) # 0.001
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--network', type=str, default='efficientnet-b5',
                        help='networks: {resnet-34, resnet-50, efficientnet-b1, '
                             'efficientnet-b5, vit-t16, vit-s16, swinv1, swinv2, convit-s， vision-mamba}')
    args = parser.parse_args()

    root = args.root
    batch_size = args.batchsize

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    writer = SummaryWriter(log_dir='runs/{}'.format(args.network))

    train_dataset, val_dataset = GetTrainingData(root, args)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    print('Train batches:', len(train_loader))
    print('Validation batches:', len(val_loader))

    logging.info(f'''Starting training:
                             Batch size:      {args.batchsize}
                             Learning rate:   {args.base_lr}
                             Dataset size:    {len(train_loader) + len(val_loader)}
                             Training size:   {len(train_loader)}
                             Validation size: {len(val_loader)}''')

    if args.network == 'resnet-18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 5)
    elif args.network == 'resnet-34':
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 5)
    elif args.network == 'resnet-50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 5)
    elif args.network == 'efficientnet-b1':
        model = torchvision.models.efficientnet_b1(pretrained=True)
        model.classifier = nn.Linear(1280, 5)
    elif args.network == 'efficientnet-b5':
        model = torchvision.models.efficientnet_b5(pretrained=True)
        model.classifier = nn.Linear(2048, 5)
    elif args.network == 'vit-t16':
        model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=False,
                                  checkpoint_path='./networks/vit_tiny_patch16_224.bin')
        model.reset_classifier(num_classes=5)
    elif args.network == 'vit-s16':
        model = timm.create_model('vit_small_patch16_224.augreg_in1k', pretrained=False,
                                  checkpoint_path='./networks/vit_small_patch16_224.bin')
        model.reset_classifier(num_classes=5)
    elif args.network == 'swinv1':
        pretrained_cfg = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k').default_cfg
        pretrained_cfg['file'] = './networks/swin_tiny_patch4_window7_224.bin'
        model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True,
                                  pretrained_cfg=pretrained_cfg)
        model.reset_classifier(num_classes=5)
    elif args.network == 'convit-s':
        pretrained_cfg = timm.create_model('convit_small.fb_in1k').default_cfg
        pretrained_cfg['file'] = './networks/convit_small.bin'
        model = timm.create_model('convit_small.fb_in1k', pretrained=True,
                                  pretrained_cfg=pretrained_cfg)
        model.reset_classifier(num_classes=5)
    elif args.network == 'swinv2':
        pretrained_cfg = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k').default_cfg
        pretrained_cfg['file'] = './networks/swin_tiny_patch4_window7_224.bin'
        model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True,
                                  pretrained_cfg=pretrained_cfg)
        model.reset_classifier(num_classes=5)

    _input = torch.rand(1, 3, 224, 224)
    flops, params = profile(model, inputs=(_input,))
    print(f'Model parameters: {params / 1e6:.2f}M')
    print(f'FLOPs: {flops / 1e9:.2f}G')
    model = model.to('cuda')

    def train(model, train_loader, val_loader, epoch, is_val=True, device='cuda'):
        if args.network in ['vit-t16', 'vit-s16', 'swinv1', 'swinv2', 'convit-s']:
            params_other_layers = [param for name, param in model.named_parameters() if
                                   name.startswith('head') == False]
            # params_other_layers_name = [name for name, param in model.named_parameters() if
            #                        name.startswith('head') == True]
            optimizer = torch.optim.Adam([
                {'params': params_other_layers, 'lr': 0.1 * args.base_lr},
                {'params': model.head.parameters(), 'lr': args.base_lr}],
                weight_decay=0.0001,
            )
        elif args.network in ['efficientnet-b1', 'efficientnet-b5']:
            params_other_layers = [param for name, param in model.named_parameters() if
                                   name.startswith('classifier') == False]
            optimizer = torch.optim.Adam([
                {'params': params_other_layers, 'lr': 0.1 * args.base_lr},
                {'params': model.classifier.parameters(), 'lr': args.base_lr}],
                weight_decay=0.0001,
            )
        else:
            params_other_layers = [param for name, param in model.named_parameters() if
                                   name.startswith('fc') == False]
            optimizer = torch.optim.Adam([
                {'params': params_other_layers, 'lr': 0.1 * args.base_lr},
                {'params': model.fc.parameters(), 'lr': args.base_lr}],
                weight_decay=0.0001,
            )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        # lf = lambda x: (((1 + math.cos(x * math.pi / epoch)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        loss_f = torch.nn.CrossEntropyLoss()

        model = model.train()
        for e in range(epoch):
            running_loss = 0
            correct = 0
            total = 0
            print(f'Start {e + 1} times training...')
            for X, y in tqdm(train_loader):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                output = model(X)
                loss = loss_f(output, y)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                running_loss += loss.item()
            running_loss /= len(train_loader)
            print('Training loss: ', running_loss)
            print('Training acc: ', correct / total)
            scheduler.step()
            print('Current lr: ', scheduler.get_last_lr())

            if is_val and (e + 1) % 5 == 0:
                acc, auc, loss = validation(model, val_loader, loss_f, device)
                print(f'The {e + 1} epoch validation accuracy is: {acc}')
                print(f'The {e + 1} epoch validation auc is: {auc}')
                print(f'The {e + 1} epoch validation loss is: {loss}')
                writer.add_scalar('Validation Acc', acc, e + 1)
                writer.add_scalar('Validation Auc', auc, e + 1)
                writer.add_scalar('Validation Loss', loss, e + 1)
                logging.info('Validation Acc: {}'.format(acc))
                logging.info('Validation Auc: {}'.format(auc))
                if (e + 1) >= 5:
                    model_path = os.path.join(args.output, f"{args.network}_{e + 1}_acc{round(acc, 4)}_auc{round(auc, 4)}.pth")
                    torch.save(model.state_dict(), model_path)

    def validation(network, loader, loss_f, device):
        network.eval()
        predictions = []
        labels = []
        loss = 0
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device)
                target = target.to(device)
                outputs = network(data)
                loss += loss_f(outputs, target)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.tolist())
                labels.extend(target.tolist())
        network.train()
        loss /= len(loader)
        labels = F.one_hot(torch.tensor(labels), num_classes=5).numpy()
        predictions = F.one_hot(torch.tensor(predictions), num_classes=5).numpy()
        accuracy = accuracy_score(labels, predictions)
        auc = roc_auc_score(labels, predictions, multi_class='ovo')
        return accuracy, auc, loss.item()

    train(model, train_loader, val_loader, epoch=40, is_val=True, device='cuda')

