
import pickle


if __name__ == "__main__":
    data_save_path="D:/Code/pycharm/Sequence-Label-Attention/attention/count_data/trigger_count.data"
    data_f = open(data_save_path, 'rb')
    pred,tag,attention_weights,L_test,words = pickle.load(data_f)
    data_f.close()
