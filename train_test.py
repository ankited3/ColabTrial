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