""" p-value computation script

    Computes p-value of the following evaluation metrics:
        %-in-test-1
        %-in-test-2
        %-in-test-3
        %-in-test-4
        BLEU-1
        BLEU-2
        BLEU-3
        BLEU-4
        
"""

from scipy import stats
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default = "./logs/", help="Directory of .npy files of both models")
parser.add_argument("--suffix1", default = "gan", help="Suffix for data files of first model")
parser.add_argument("--suffix2", default = "mle", help="Suffix for data files of second model")

args = parser.parse_args()

# Load data

tot_unigrams_suffix1 = np.load(args.data_path+"tot_unigrams_"+args.suffix1+".npy")
tot_bigrams_suffix1 = np.load(args.data_path+"tot_bigrams_"+args.suffix1+".npy")
tot_trigrams_suffix1 = np.load(args.data_path+"tot_trigrams_"+args.suffix1+".npy")
tot_quadgrams_suffix1 = np.load(args.data_path+"tot_quadgrams_"+args.suffix1+".npy")
tot_bleu1_suffix1 = np.load(args.data_path+"tot_bleu1_"+args.suffix1+".npy")
tot_bleu2_suffix1 = np.load(args.data_path+"tot_bleu2_"+args.suffix1+".npy")
tot_bleu3_suffix1 = np.load(args.data_path+"tot_bleu3_"+args.suffix1+".npy")
tot_bleu4_suffix1 = np.load(args.data_path+"tot_bleu4_"+args.suffix1+".npy")

tot_unigrams_suffix2 = np.load(args.data_path+"tot_unigrams_"+args.suffix2+".npy")
tot_bigrams_suffix2 = np.load(args.data_path+"tot_bigrams_"+args.suffix2+".npy")
tot_trigrams_suffix2 = np.load(args.data_path+"tot_trigrams_"+args.suffix2+".npy")
tot_quadgrams_suffix2 = np.load(args.data_path+"tot_quadgrams_"+args.suffix2+".npy")
tot_bleu1_suffix2 = np.load(args.data_path+"tot_bleu1_"+args.suffix2+".npy")
tot_bleu2_suffix2 = np.load(args.data_path+"tot_bleu2_"+args.suffix2+".npy")
tot_bleu3_suffix2 = np.load(args.data_path+"tot_bleu3_"+args.suffix2+".npy")
tot_bleu4_suffix2 = np.load(args.data_path+"tot_bleu4_"+args.suffix2+".npy")

# Print p-value

_, p =stats.ttest_ind(tot_unigrams_suffix1,tot_unigrams_suffix2)
print ("%%-in-test-1 p-value: %f" % p)
_, p =stats.ttest_ind(tot_bigrams_suffix1,tot_bigrams_suffix2)
print ("%%-in-test-2 p-value: %f" % p)
_, p =stats.ttest_ind(tot_trigrams_suffix1,tot_trigrams_suffix2)
print ("%%-in-test-3 p-value: %f" % p)
_, p =stats.ttest_ind(tot_quadgrams_suffix1,tot_quadgrams_suffix2)
print ("%%-in-test-4 p-value: %f" % p)
_, p =stats.ttest_ind(tot_bleu1_suffix1,tot_bleu1_suffix2)
print ("bleu1 p-value: %f" % p)
_, p =stats.ttest_ind(tot_bleu2_suffix1,tot_bleu2_suffix2)
print ("bleu2 p-value: %f" % p)
_, p =stats.ttest_ind(tot_bleu3_suffix1,tot_bleu3_suffix2)
print ("bleu3 p-value: %f" % p)
_, p =stats.ttest_ind(tot_bleu4_suffix1,tot_bleu4_suffix2)
print ("bleu4 p-value: %f" % p)
