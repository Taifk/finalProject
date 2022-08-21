import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from torch import nn, optim
from PIL import Image
import os, random

def data_transformation(args) :
    
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")
    
    if not os.path.exists(args.data_directory):
        
        print("Data Directory doesn't exist: {}".format(args.data_directory))
        
        raise FileNotFoundError
        
    if not os.path.exists(args.save_directory):
        
        print("Save Directory doesn't exist: {}".format(args.save_directory))
        
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        
        print("Train folder doesn't exist: {}".format(train_dir))
        
        raise FileNotFoundError
        
    if not os.path.exists(valid_dir):
        
        print("Valid folder doesn't exist: {}".format(valid_dir))
        
        raise FileNotFoundError
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    valid_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    
    train_data = ImageFolder(root = train_dir, transform = train_transforms)
    valid_data = ImageFolder(root = valid_dir, transform = valid_transforms)

    train_loader = data.DataLoader(train_data, batch_size = 64, shuffle = True)
    valid_loader = data.DataLoader(valid_data, batch_size = 64, shuffle = True)
    
    return train_loader, valid_loader, train_data.class_to_idx

def train_model(args, train_loader, valid_loader, class_to_idx) :
    
    if args.model_arch == "vgg11":
        
        model = torchvision.models.vgg11(pretrained=True)
        
    elif args.model_arch == "vgg13":
        
        model = torchvision.models.vgg13(pretrained=True)
        
    elif args.model_arch == "vgg16":
        
        model = torchvision.models.vgg16(pretrained=True)
        
    elif args.model_arch == "vgg19":
        
        model = torchvision.models.vgg19(pretrained=True)
        
    for param in model.parameters() :
        
        param.requires_grad = False
        
    features_of_pretrained = model.classifier[0].in_features
    
    classifier = nn.Sequential(nn.Linear(in_features = features_of_pretrained, out_features = 2048, bias = True),
                               nn.ReLU(inplace = True),
                               nn.Dropout(p = 0.2),
                               nn.Linear(in_features = 2048, out_features = 102, bias = True),
                               nn.LogSoftmax(dim = 1)
                              )
    
    model.classifier = classifier
     
    criterion = nn.NLLLoss()
        
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    
    if args.gpu and torch.cuda.is_available():
            
        device = 'cuda'
        
    elif args.gpu and not(torch.cuda.is_available()):
        
        device = 'cpu'
        
        print("GPU was selected as the training device, but no GPU is available. Using CPU instead")
    else:
        device = 'cpu'
    print("Using {} to train model".format(device))
    
    model.to(device)
        
    every = 20
    
    for e in range(args.epochs) :
        
        step = 0
        running_train_loss = 0
        running_valid_loss = 0
        
        for inputs, labels in train_loader :
            
            step += 1
            
            model.train()

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            train_loss = criterion(outputs, labels)

            train_loss.backward()

            optimizer.step()

            running_train_loss += train_loss.item()
            
            if step % every == 0 or step == 1 or step == len(train_loader):
                
                print("Epoch: {}/{} Batch % Complete: {:.2f}%".format(e + 1, args.epochs, (step) * 100 / len(train_loader)))
                
        model.eval()  
        
        with torch.no_grad() :
            
            print("Validating Epoch :")
            
            running_accuracy = 0
            running_valid_loss = 0
            
            for inputs, labels in valid_loader :
                
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                valid_loss = criterion(outputs, labels)
                
                running_valid_loss += valid_loss.item()
                
                ca = torch.exp(outputs)
                    
                top_p, top_class = ca.topk(1, dim = 1)
                
                equals = top_class == labels.view(*top_class.shape)
                
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            average_train_loss = running_train_loss / len(train_loader)
                    
            average_valid_loss = running_valid_loss / len(valid_data_loader)
            
            accuracy = running_accuracy / len(valid_loader)
            
            print("Train Loss: {:.3f}".format(average_train_loss))
            print("Valid Loss: {:.3f}".format(average_valid_loss))
            print("Accuracy: {:.3f}%".format(accuracy * 100))
            
     model.class_to_idx = class_to_idx
    
     checkpoint = {'classifier' : model.classifier,
                  'state_dict' : model.state_dict(),
                  'epochs' : args.epochs,
                  'optim_stat_dict' : optimizer.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'vgg_type' : args.model_arch
                 }
     
     torch.save(checkpoint, os.path.join(args.save_directory, "checkpoint.pth"))
        
     print("model saved to {}".format(os.path.join(args.save_directory, "checkpoint.pth")))
    
     return True
    
if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(dest = 'data_directory', help = "This is the dir of the training images. Expect 2 folders within, 'train' & 'valid'")
    
    args = parser.parse_args()
    
    train_loader, valid_data_loader, class_to_idx = data_transformation(args)
        
    train_model(args, train_loader, valid_loader, class_to_idx)
        