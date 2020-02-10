import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.nn.functional as F

a = torch.ones(3, 3)  # tensor obsahuju jednotky
print(a)  # zobrazenie tensoru

b = torch.ones(3, 3)  # tensor obsahuju jednotky
c = a + b  # scitanie tensorov
print(c)

num_epochs = 10              #pocet epoch
batch_size = 32              #velkost vzorky
learning_rate = 0.001
log_interval = 10
device = torch.device('cuda')


class Net(nn.Module):					#definicia triedy Net(nasej neuronovej siete)
    def __init__(self):					# a jednotlivych vrstiev
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)		#konvolucna vrstva 1 
        self.conv2 = nn.Conv2d(32, 64, 3, 1)		#konvolucna vrstva 2 
        self.dropout1 = nn.Dropout2d(0.25) 		#dropout vrstva 1 
        self.dropout2 = nn.Dropout2d(0.5)		#dropout vrstva 2 
        self.fc1 = nn.Linear(9216, 128)			#plne prepojena vrstva 1 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):				#definicia funkcie forward() kde sa 
        x = self.conv1(x)				#vrstvy skladaju na seba a pridaju sa aktivacne funkcie
        x = F.relu(x)					#pridanie aktivacnej funkcie k relu vrstve
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):		# definicia funkcie train ktoretrenovacia faza 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()						# vynulovanie gradientu
        output = model(data)
        loss = F.nll_loss(output, target)				# urcenie nulovej funkcie
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():                               # V testovacej faze sa nepocitaju gradienty
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# MNIST dataset
# Definovanie trenovacej casti dat
train_dataset = datasets.MNIST(root='data',				# definovanie trenovacej casti dat
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)				# ak nie su stiahnute data tak sa stiahnu
# Definovanie testovacej casti dat
test_dataset = datasets.MNIST(root='data',				# definovanie testovacej casti dat
                              train=False,
                              transform=transforms.ToTensor())

# Nacitanie dat
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,	# nacitanie trenovacich dat
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,		# nacitanie testovacich dat
                                          batch_size=batch_size,
                                          shuffle=False)

model = Net()								# vytvorenie instancie siete
print(model)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)		# priradenie optimalizacneho algoritmu
model.to(device)							# priradenie modelu na graficku kartu

for epoch in range(1, num_epochs + 1):					# cyklus iduci od 1 po pocet epoch+1
    train(model, device, train_loader, optimizer, epoch)		# volanie funkcie train
    test(model, device, test_loader)					# volanie funkcie test
