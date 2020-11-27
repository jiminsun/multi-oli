from data_utils.utils import load_data, save_data
import argparse


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str)
    p.add_argument('-n', '--num_samples', type=int)
    args = p.parse_args()

    df = load_data('../results/lang/en_train_da_mbert-epoch=10-val_loss=0.463-val_f1=0.767.tsv')
    # Select samples with lowest english score
    df = df.sort_values(by='logit', ascending=True)
    print(len(df))
    for n in [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, len(df)]:
        sample = df.head(n)
        out_fname = f'../dataset/en_transferable/train_{n}.txt'
        save_data(sample, out_fname)