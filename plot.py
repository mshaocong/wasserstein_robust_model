import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 18})


max_length = 2500

with open("result-momentum.pkl", "rb") as fp:  # Pickling
    result_momentum = pickle.load(fp)[:max_length]
fp.close()
with open("result-raw.pkl", "rb") as fp:  # Pickling
    result_raw = pickle.load(fp)[:max_length]
fp.close()
with open("result-apda.pkl", "rb") as fp:  # Pickling
    result_2 = pickle.load(fp)[:max_length]
fp.close()
with open("result-ALT.pkl", "rb") as fp:  # Pickling
    result_momentum2 = pickle.load(fp)[:max_length]
fp.close()

with open("acc-momentum.pkl", "rb") as fp:  # Pickling
    acc_1 = pickle.load(fp)[:max_length]
fp.close()
with open("acc-raw.pkl", "rb") as fp:  # Pickling
    acc_2 = pickle.load(fp)[:max_length]
fp.close()
with open("acc-apda.pkl", "rb") as fp:  # Pickling
    acc_3 = pickle.load(fp)[:max_length]
fp.close()
with open("acc-ALT.pkl", "rb") as fp:  # Pickling
    acc_4 = pickle.load(fp)[:max_length]
fp.close()



fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.plot(np.array(result_momentum).squeeze(), c="tab:orange",  label="Proximal AltGDA", linestyle="dotted", alpha=0.7, linewidth=6)
ax1.plot(np.array(result_raw).squeeze(), c="tab:blue",   label="Proximal GDA", linestyle="dashed", linewidth=2 )
ax1.plot(np.array(result_2).squeeze(), c="tab:green",   label="APDA", linestyle="dashdot", linewidth=3)
ax1.plot(np.array(result_momentum2).squeeze(), c="tab:red",   label="Proximal AltGDAm")
ax1.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': 14})
ax1.set_xlabel("# of epochs")
ax1.set_ylabel(r"Estimated $\Phi(x)+g(x)$")
# ax1.set_yscale('log')
ax1.patch.set_facecolor('white')
fig1.patch.set_facecolor('white')
fig1.tight_layout()
plt.savefig("MNIST-loss.png" , dpi=300,  facecolor="white", edgecolor='none')
# plt.show()

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(acc_1, c="tab:orange", label="Proximal AltGDA", linestyle="dotted", alpha=0.7, linewidth=6)
ax2.plot(acc_2, c="tab:blue", label="Proximal GDA", linestyle="dashed", linewidth=2 )
ax2.plot(acc_3, c="tab:green", label="APDA", linestyle="dashdot", linewidth=3 )
ax2.plot(acc_4, c="tab:red", label="Proximal AltGDAm")
ax2.legend(loc='upper left', fancybox=True, shadow=True, prop={'size': 14} )
ax2.set_xlabel("# of epochs")
ax2.set_ylabel(r"Test accuracy")
ax2.patch.set_facecolor('white')
fig2.patch.set_facecolor('white')
fig2.tight_layout()
plt.savefig("MNIST-acc.png" , dpi=300, facecolor="white", edgecolor='none')
# plt.show()