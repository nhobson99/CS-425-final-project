import numpy as np
import pandas as pd
import sys
import gf
import math

if (len(sys.argv) == 1):
    print("Usage: python3 linreg3.py input_file [--squares --cubes]")
    exit(-1)

filename = sys.argv[1]

# If squares, append square values of each feature to the list
# If logs, append logarithmic scale of each feature to feature list

extensions = {
    "--squares" : (lambda x : x**2),
    "--invs"    : (lambda x : x**-1),
    "--cubes"   : (lambda x : x**3),
    "--sqrts"   : (lambda x : x**-2),
    "--logs"    : (lambda x : math.log(x))
}

for ext in list(extensions.keys()):
    if (not ext in sys.argv):
        extensions.pop(ext)

# The higher, the better

def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# Apply the weight function to input x
def f(w, x):
    if (len(w) != len(x)):
        print("Need same size w and x, got " +
              str(len(w)) + "," + str(len(x)))
        exit(-1)
    rv = np.dot(w, x)
    # clip to range [1, 5]
    # convert to integers for clarity
    return np.clip(rv, 1, 5).astype(int)


# Runs the function (f) with weights (w) on every data point (x)
def Predict(w, features):
    Yp = []
    for x in features:
        Yp.append(f(w, x))
    return (np.array(Yp))


data = pd.read_csv(filename)


def linreg(expected, features):
    global extensions
    x = [list(data[feat][1:]) for feat in features]

    for ext in extensions:
        for feat in features:
            x.append([extensions[ext](float(n)) for n in list(data[feat][1:])])

    x.append([1]*len(x[0]))

    xT = np.array(x, dtype=float)
    x = xT.T

    Y = np.array(data[expected][1:], dtype=float)       # Expected output

    # Solve the actual linear regression
    w = np.matmul(np.linalg.inv(np.matmul(xT, x)), np.matmul(xT, Y))

    Yp = Predict(w, x)

    # print("\tWeights (w): " + str(w))
    print("\tR² = " + str(r2_score(Y, Yp)))
    # print("\tAverage distance from correct: " + str(np.average(Y-Yp)))

    return Y, Yp, w


print("\nCommit size:")
features_c = ["num_files_changed", "num_lines_changed"]
expected_c = "rank_commit_size"
rank_commit_size, rank_commit_size_p, weights_c = linreg(expected_c, features_c)
percent_correct_c = np.sum(rank_commit_size == rank_commit_size_p) / len(rank_commit_size)
print("\tPercent correct: " + str(percent_correct_c))

print("\nMessage length:")
features_l = ["gunning_fog_approx", "num_chars", "num_words", "chars_per_word", "num_sentences", "words_per_sentence"]
expected_l = "rank_message_length"
rank_message_length, rank_message_length_p, weights_l = linreg(expected_l, features_l)
percent_correct_l = np.sum(rank_message_length == rank_message_length_p) / len(rank_message_length)
print("\tPercent correct: " + str(percent_correct_l))

print("\nMessage readability")
features_r = ["gunning_fog_approx", "num_chars", "num_words", "chars_per_word", "num_sentences", "words_per_sentence"]
expected_r = "rank_message_readability"
rank_message_readability, rank_message_readability_p, weights_r = linreg(expected_r, features_r)
percent_correct_r = np.sum(rank_message_readability == rank_message_readability_p) / len(rank_message_readability)
print("\tPercent correct: " + str(percent_correct_r))

print()

csv_output = [rank_commit_size, rank_message_length, rank_message_readability,
              list(data["rank_commit_size"][1:]),
              list(data["rank_message_length"][1:]),
              list(data["rank_message_readability"][1:])]

csv_output = list(np.array(csv_output).T)

pd.DataFrame(csv_output).to_csv("ranks.csv", index=False, header=False, float_format="%.0f")

while True:
    try:
        s = input("Enter commit message: ")
    except EOFError:
        print()
        break

    _gf = gf.gunning_fog(s)
    num_chars = len(s)
    num_words = len(s.split())
    try:
        chars_per_word = num_chars / num_words
    except:
        chars_per_word = num_chars
    num_sentences = len(s.split('.'))
    try:
        words_per_sentence = num_words / num_sentences
    except:
        words_per_sentence = 0
    feats = [_gf, num_chars, num_words, chars_per_word, num_sentences, words_per_sentence]
    for ext in extensions:
        for feat in [_gf, num_chars, num_words, chars_per_word, num_sentences, words_per_sentence]:
            try:
                feats.append(extensions[ext](feat))
            except:
                feats.append(feat)

    feats.append(1)

    test = np.array(feats)
    print("Readability: " + str(f(weights_r, test)))
    print("Message length: " + str(f(weights_l, test)))