import matplotlib.pyplot as plt 
import pickle

with open('plot/plot.pkl','rb') as file:
	tr, val=pickle.load(file)

plt.plot(tr, label='Training')
plt.plot(val,label='Validation')
plt.legend(loc='lower right')
plt.xlabel('Number of epochs ->')
plt.ylabel('Accuracy ->')
plt.show()