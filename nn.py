from torch import nn,save, load
from torch.optim import Adam            # using adam optimizer
from torch.utils.data import DataLoader 
from torchvision import datasets             #using the pytorch dataset on image detection
from torchvision.transforms import ToTensor

#converting the data to tensors and loading images in sets of 32
train=datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset=DataLoader(train, 32)

#Layers of the neural network
class ImageClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.model=nn.Sequential(
		nn.Conv2d(1,32,(3,3)),
		nn.ReLU(),
		nn.Conv2d(32,64,(3,3)),
		nn.ReLU(),
		nn.Conv2d(64,64,(3,3)),
		nn.ReLU(),
		nn.Flatten(),
		nn.Linear(64*(22)*(22),10)

		)
	def forward(self,x):
		return self.model(x)
# imtance of classifier, optimizer and loss function
clf=ImageClassifier().to('cuda') #to "cude" for gpu 
opt=Adam(clf.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()

 # e is epoch since we have 10 epochs
if __name__ =="__main__":
	for e in range(10):
		for batch in dataset:
			X,y=batch
			X,y=X.to('cuda'),y.to('cuda') #'cuda' for GPU and 'cpu' for cpu
			yhat=clf(X)
			loss=loss_fn(yhat,y)

			opt.zero_grad()
			loss.backward()
			opt.step()

		
#training the neural network
		print(f"Epoch:{e} loss is {loss.item()}")
	with open('model_state.pt', 'wb') as f:	
		save(clf.state_dict(),f)



