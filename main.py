from optimizer import *

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)


if __name__ == '__main__':
    batch_size = 1000
    lr1 = 0.001
    lr2 = 0.001
    momentum1 = 0.1
    momentum2 = 0.1
    gamma = 1.3
    num_epochs = 30
    result_2, acc_3 = gda2(batch_size, lr1, lr2, lr1, gamma, num_epochs)
    result_momentum, acc_1 = gda(batch_size, lr1, lr2, gamma, num_epochs, momentum1, momentum2)
    #result_momentum2 = gda(batch_size, lr1, lr2, gamma, num_epochs, momentum1 * 10, momentum2 * 10)
    #result_momentum3 = gda(batch_size, lr1, lr2, gamma, num_epochs, momentum1 * 100, momentum2 * 100)
    result_raw, acc_2 = gda(batch_size, lr1, lr2, gamma, num_epochs, 0.0, 0.0)

    #result_2, acc_3 = gda2(batch_size, lr1, lr2, lr1, gamma, num_epochs)

    import matplotlib.pyplot as plt
    import pickle
    with open("test-momentum.pkl", "wb") as fp:  # Pickling
        pickle.dump(result_momentum, fp)
    fp.close()
    # with open("test-momentum2.pkl", "wb") as fp:  # Pickling
    #     pickle.dump(result_momentum2, fp)
    # fp.close()
    # with open("test-momentum3.pkl", "wb") as fp:  # Pickling
    #     pickle.dump(result_momentum3, fp)
    # fp.close()
    with open("test-raw.pkl", "wb") as fp:  # Pickling
        pickle.dump(result_raw, fp)
    fp.close()
    with open("test-2.pkl", "wb") as fp:  # Pickling
        pickle.dump(result_2, fp)
    fp.close()


    #plt.plot(np.array(result_momentum3).squeeze(), c="r")
    #plt.plot(np.array(result_momentum2).squeeze(), c="orange")
    plt.plot(np.array(result_momentum).squeeze(), c="r", label="Momentum1="+str(momentum1)+" Momentum2="+str(momentum2))
    plt.plot(np.array(result_raw).squeeze(), c="b", label="GDA without momentum")
    plt.plot(np.array(result_2).squeeze(), c="g", label="Accelerated GDA")
    plt.legend()
    plt.show()


    # plt.plot(acc_1, c="r")
    # plt.plot(acc_2, c="b")
    # plt.plot(acc_3, c="g")
    # plt.show()