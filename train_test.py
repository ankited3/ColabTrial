from model1_test import (
    model as model1,
    device as device1,
    train_loader as train_loader1,
    test_loader as test_loader1,
    train as train1,
    test as test1,
    train_losses as train_losses1,
    train_acc as train_acc1,
    test_losses as test_losses1,
    test_acc as test_acc1
)
from model2_test import (
    model as model2,
    device as device2,
    train_loader as train_loader2,
    test_loader as test_loader2,
    train as train2,
    test as test2,
    train_losses as train_losses2,
    train_acc as train_acc2,
    test_losses as test_losses2,
    test_acc as test_acc2
)
import torch.optim as optim


optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
EPOCHS = 2
for epoch in range(EPOCHS):
    print("Model 1 EPOCH:", epoch)
    train1(model1, device1, train_loader1, optimizer1, epoch)
    test1(model1, device1, test_loader1)

t = [t_items.item() for t_items in train_losses1]

# ... plotting code ...

optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
EPOCHS = 2
for epoch in range(EPOCHS):
    print("Model 2 EPOCH:", epoch)
    train2(model2, device2, train_loader2, optimizer2, epoch)
    test2(model2, device2, test_loader2)