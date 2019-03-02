from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)

        #add a batchnorm layer
        self.batchnorm_1 = nn.BatchNorm1d(512, 1e-12, affine=True, track_running_stats=True)

        #add a fully-connected layer before the FC2 layer

        self.fc_new = nn.Linear(512,512)

        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)

        # print("x ",x.shape)

        x = self.batchnorm_1(x)#--------------added

        x = F.relu(x)

        x = self.fc_new(x)#-------------------added


        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # ----------------to change into evaluation mode
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.data
    net.train() # Why would I do this?



    return total_loss.item() / total, correct.item() / total

if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train

    #--------------Load the saved weights on a dictionary
    saved_state_statistics_of_the_model = torch.load("mytraining1b.pth")
    for keyname_of_the_state_statistic in saved_state_statistics_of_the_model:

        print(keyname_of_the_state_statistic )


    #-------Since you do not want to use all of the weights but only those weights before the FC2 layer,
    # copy the dictionary into some other variable,
    # then remove unnecessary weights
    #from this new dictionary and then use the remaining wights to apply to the model

    copy_of_saved_state_statistics_of_the_model = saved_state_statistics_of_the_model.copy()
    #Now once we initialize the weights for all the layers, we use this new copy of the pretrained weights to overwrite those initializations as we require




    print("Current cuda device is ", torch.cuda.current_device())
    print("Total cuda-supporting devices count is ",torch.cuda.device_count())
    print("Current cuda device name is ",torch.cuda.get_device_name(torch.cuda.current_device()))




    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean for the color channels, standard deviation for the color channels)

    #Note that by default CIFAR10 is a colored image dataset
    trainset = torchvision.datasets.CIFAR10(root='../../../data', train=True,
                                            download=True, transform=transform)

    # print("trainset ",trainset) #returns information about the dataset that you put into the variable called "trainset", even displays the information about what you did on this data
    # print("trainset [0]", trainset[0])# is a tuple of the form ( tensor of the shape [3, 32, 32], 6) which is basically features, label
    # print("trainset [0] [0] ",trainset[0][0].shape) #[3, 32, 32], each image has 3 channels
    # print("trainset [0] [1] ",trainset[0][1])#prints the label for this image


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2) #shuffle = True shuffles data at every new epoch

    testset = torchvision.datasets.CIFAR10(root='../../../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda() #VVI: Always move the model to GPU before constructing an optimizer, it doesnt matter if you are using SGD as an optimizer but you will not get the efficiency you wsant if you dont
    #do this for other optimizers

    # ---------Lets initialize the weights partially before the training
    model_statistics_dictionary = net.state_dict()  # get the model's statistics first
    del copy_of_saved_state_statistics_of_the_model["fc2.weight"]  # ---------------delete the weights/ biases/ statistics that you don't want to use to update your model with
    del copy_of_saved_state_statistics_of_the_model["fc2.bias"]
    del copy_of_saved_state_statistics_of_the_model["batchnorm_1.running_mean"]
    del copy_of_saved_state_statistics_of_the_model["batchnorm_1.running_var"]
    del copy_of_saved_state_statistics_of_the_model["batchnorm_1.num_batches_tracked"]


    # -----------------Now update the model's weights/ biases for those keys that exit in this edited dictionary of weights and biases


    for key, value in copy_of_saved_state_statistics_of_the_model.items():
        if key in model_statistics_dictionary:
            model_statistics_dictionary.update({key: value})  # -------------------------------------


    # -------------------------------->Now load these parameters to the model from the variable
    net.load_state_dict(model_statistics_dictionary)

    net.train() # Why would I do this? -------> sets the module in training mode
    #train() is a function defined for nn.Module() class. It sets the module in training mode.
    # #This    has  an    effect    only    on    certain    modules.See   documentations   of   particular  modules    for details of their behaviors in training / evaluation mode, if they are affected, e.g.Dropout, BatchNorm, etc.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
                            [
                                    {"params" : net.fc_new.parameters()},
                                    {"params":net.fc2.parameters()}

                            ],lr=0.01,momentum=0.9
                            )

    #---------------keep track of some variables after each epoch, plot after each epoch, and show after all the epochs are over
    epoch_list_for_the_plot = []
    training_accuracy_list = []
    testing_accuracy_list = []

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the data set multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            #print("loss.data ",loss.data)




            running_loss += loss.data
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))


        #----------------------------Append the values to the lists
        training_accuracy_list.append(train_acc)
        testing_accuracy_list.append(test_acc)
        epoch_list_for_the_plot.append(epoch)

        #----------------------------Plot the results for each epoch
        plt.plot(epoch_list_for_the_plot, training_accuracy_list, 'g', label="Training Accuracies for the Epochs")  # pass array or list
        plt.plot(epoch_list_for_the_plot, testing_accuracy_list, 'r', label="Testing Accuracies for the Epochs")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracies")
        plt.title("Number of Epochs VS Accuracies")





    print('Finished Training')
    print('Saving model...')
    plt.show()





    torch.save(net.state_dict(), 'mytraining2.pth')
