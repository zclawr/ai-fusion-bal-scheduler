import pickle
import numpy as np

if __name__ == "__main__":
    with open('../samples_top2500_1.pkl', 'rb') as file:
        inputs_chunk1 = pickle.load(file)
    with open('../samples_top2500_2.pkl', 'rb') as file:
        inputs_chunk2 = pickle.load(file)
    with open('../samples_top2500_3.pkl', 'rb') as file:
        inputs_chunk3 = pickle.load(file)
    with open('../samples_top2500_4.pkl', 'rb') as file:
        inputs_chunk4 = pickle.load(file)
    inputs_all = np.concatenate([inputs_chunk1, inputs_chunk2, inputs_chunk3, inputs_chunk4], axis=0)
    inputs_10 = inputs_all[:10, :]
    np.save('./samples_top10000.npy', inputs_all)
    np.save('./samples_10_test.npy', inputs_10)