from optimizer import *

import matplotlib.pyplot as plt
import pickle
torch.manual_seed(0)
np.random.seed(0)


if __name__ == '__main__':
    batch_size = 60000
    lr1 = 0.001
    lr2 = 0.001
    gamma = 1.3
    num_epochs = 200 #200

    momentum1 = 0.25
    momentum2 = 0.75

    lr3 = 0.001 * 2

    count = 0
    count2 = 0

    result_2, acc_3 = gda2(batch_size, lr1, lr2, lr3, gamma, num_epochs) # APDA
    result_momentum, acc_1 = gda(batch_size, lr1, lr2, gamma, num_epochs, momentum1, momentum2) # NO ALT + MOMENTUM
    result_raw, acc_2 = gda(batch_size, lr1, lr2, gamma, num_epochs, 0.0, 0.0) # NO ALT + NO MOMENTUM
    result_momentum2, acc_4 = gda3(batch_size, lr1, lr2, gamma, num_epochs, momentum1, momentum2) # ALT + MOMENTUM

    plt.figure()
    plt.plot(np.array(result_momentum).squeeze(), c="r", label="GDAm")
    plt.plot(np.array(result_raw).squeeze(), c="b", label="GDA")
    plt.plot(np.array(result_2).squeeze(), c="g", label="APDA")
    plt.plot(np.array(result_momentum2).squeeze(), c="orange", label="ALT")
    plt.legend()
    plt.savefig("fig_loss_MNIST.png", dpi=300)

    plt.figure()
    plt.plot(acc_1, c="r", label="GDAm")
    plt.plot(acc_2, c="b", label="GDA")
    plt.plot(acc_3, c="g", label="APDA")
    plt.plot(acc_4, c="orange", label="ALT")
    plt.legend()
    plt.savefig("fig_acc_MNIST.png", dpi=300)


    with open("result-momentum.pkl", "wb") as fp:  # Pickling
        pickle.dump(result_momentum, fp)
    fp.close()
    with open("result-raw.pkl", "wb") as fp:  # Pickling
        pickle.dump(result_raw, fp)
    fp.close()
    with open("result-apda.pkl", "wb") as fp:  # Pickling
        pickle.dump(result_2, fp)
    fp.close()
    with open("result-ALT.pkl", "wb") as fp:  # Pickling
        pickle.dump(result_momentum2, fp)
    fp.close()
    with open("acc-momentum.pkl", "wb") as fp:  # Pickling
        pickle.dump(acc_1, fp)
    fp.close()
    with open("acc-raw.pkl", "wb") as fp:  # Pickling
        pickle.dump(acc_2, fp)
    fp.close()
    with open("acc-apda.pkl", "wb") as fp:  # Pickling
        pickle.dump(acc_3, fp)
    fp.close()
    with open("acc-ALT.pkl", "wb") as fp:  # Pickling
        pickle.dump(acc_4, fp)
    fp.close()