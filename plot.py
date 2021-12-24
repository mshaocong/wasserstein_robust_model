import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})



with open("accuracy_FashionMNIST.pickle", "rb") as fp:  # Pickling
    accuracy = pickle.load(fp)
fp.close()

acc_vanilla = accuracy["vanilla"]
acc_momentum = accuracy["momentum"]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(acc_vanilla, c="tab:orange", label="Vanilla", linestyle="--", linewidth=4)
ax.plot(acc_momentum, c="tab:blue", label="Momentum",  linewidth=4)
ax.legend(loc='lower right', fancybox=True, shadow=True, prop={'size': 18})
ax.set_xlabel("# of epochs")
ax.set_ylabel(r"Test accuracy")
ax.patch.set_facecolor('white')
fig.patch.set_facecolor('white')
fig.suptitle('FashionMNIST Dataset', fontsize=24)
fig.tight_layout()
plt.savefig("FashionMNIST-acc.png" , dpi=300, facecolor="white", edgecolor='none')
# plt.show()