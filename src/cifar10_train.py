from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
FLAGS = flags.FLAGS


class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 8, 1)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2)
        self.fc = nn.Linear(128 * 3 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc(x)
        return x


def ld_cifar10():
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="/tmp/data", train=True, transform=train_transforms, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="/tmp/data", train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


def main(_):
    # Load training and test data
    data = ld_cifar10()

    # Instantiate model, loss, and optimizer for training
    net = CNN(in_channels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    univ_pert = torch.zeros(1, 3, 32, 32).to(device)


    # Train vanilla model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )

    # develop universal perturbation for the dataset using the net
    epsilon = 0.0625
    eps_iter = 0.01
    # Initialize universal perturbation
    univ_pert = torch.zeros(1, 3, 32, 32).to(device)
    univ_pert_reg = torch.zeros(1, 3, 32, 32).to(device)

    # Hyperparameters for universal perturbation
    epsilon = 0.0625
    eps_iter = 0.01
    num_epochs = 20

    channels = 3
    kernel_size = 5
    ws = torch.ones((channels, 1, kernel_size, kernel_size)) /(kernel_size**2)
    ws = ws.to(device)


    net.eval()
    for epoch in range(1, num_epochs + 1):
        total_grad = torch.zeros_like(univ_pert)
        train_loss = 0.0
        num_batches = 0
        
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            
            # Apply current universal perturbation
            x_perturbed = x + univ_pert
            x_perturbed = torch.clamp(x_perturbed, 0, 1)  # Ensure valid image range
            
            # Compute loss and gradients
            x_perturbed.requires_grad = True
            loss = loss_fn(net(x_perturbed), y)
            loss.backward()
            
            # Accumulate gradients
            total_grad += x_perturbed.grad.sum(dim=0, keepdim=True)
            
            train_loss += loss.item()
            num_batches += 1
        
        # Update universal perturbation
        univ_pert = univ_pert + eps_iter * torch.sign(total_grad)
        univ_pert = torch.clamp(univ_pert, -epsilon, epsilon)
        
        print(f"Epoch: {epoch}/{num_epochs}, Average train loss: {train_loss / num_batches:.3f} train loss: {train_loss}")
    print("Shape of the universal perturbation: ", univ_pert.shape)
    print(f"L2 norm of universal perturbation: {torch.norm(univ_pert)}")    # Evaluate on clean and adversarial data


    for epoch in range(1, num_epochs + 1):
        total_grad = torch.zeros_like(univ_pert)
        train_loss = 0.0
        num_batches = 0
        
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            
            # Apply current universal perturbation
            x_perturbed = x + univ_pert_reg
            x_perturbed = torch.clamp(x_perturbed, 0, 1)  # Ensure valid image range
            
            # Compute loss and gradients
            x_perturbed.requires_grad = True
            loss = loss_fn(net(x_perturbed), y)
            response_map = torch.nn.functional.conv2d((x_perturbed - x), ws, groups=channels)
            regularization = torch.sum(torch.abs(response_map))
            loss += regularization
            loss.backward()
            
            # Accumulate gradients
            total_grad += x_perturbed.grad.sum(dim=0, keepdim=True)
            
            train_loss += loss.item()
            num_batches += 1
        
        # Update universal perturbation
        univ_pert_reg = univ_pert_reg + eps_iter * torch.sign(total_grad)
        univ_pert_reg = torch.clamp(univ_pert_reg, -epsilon, epsilon)
        
        print(f"Epoch: {epoch}/{num_epochs}, Average train loss: {train_loss / num_batches:.3f} train loss: {train_loss}")
    print("Shape of the universal perturbation: ", univ_pert.shape)
    print(f"L2 norm of universal perturbation: {torch.norm(univ_pert)}")    # Evaluate on clean and adversarial data


    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0, correct_univ_pert=0, correct_univ_pert_reg=0)
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        # x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
        # x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
        _, y_pred = net(x).max(1)  # model prediction on clean examples
        _, y_pred_univ = net(x + univ_pert).max(1)  # model prediction on adversarial examples
        _, y_pred_univ_reg = net(x + univ_pert_reg).max(1)
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_univ_pert += y_pred_univ.eq(y).sum().item()
        report.correct_univ_pert_reg += y_pred_univ_reg.eq(y).sum().item()
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    print(
        "test acc on adversarial examples (%): {:.3f}".format(
            report.correct_univ_pert / report.nb_test * 100.0
        )
    )
    print(
        "test acc on adversarial examples (%): {:.3f}".format(
            report.correct_univ_pert_reg / report.nb_test * 100.0
        )
    )
    # save the perturbations as pickle files
    with open("univ_pert.pkl", "wb") as f:
        pickle.dump(univ_pert, f)
    with open("univ_pert_reg.pkl", "wb") as f:
        pickle.dump(univ_pert_reg, f)



if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 10, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)
