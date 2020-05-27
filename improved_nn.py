import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx


def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):
    # structure of bigram probability matrix will be:
    # (last word, current word) --> probability
    # we will use add-1 smoothing
    # note: we'll always ignore this from the END token
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):

            if i == 0:
                # beginning word
                bigram_probs[start_idx, sentence[i]] += 1
            else:
                # middle word
                bigram_probs[sentence[i-1], sentence[i]] += 1

            # if we're at the final word
            # we update the bigram for last -> current
            # AND current -> END token
            if i == len(sentence) - 1:
                # final word
                bigram_probs[sentence[i], end_idx] += 1

    # normalize the counts along the rows to get probabilities
    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
    return bigram_probs


if __name__ == '__main__':
    # load in the data
    # note: sentences are already converted to sequences of word indexes
    # note: you can limit the vocab size if you run out of memory
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
    # sentences, word2idx = get_sentences_with_word2idx()

    # vocab size
    V = len(word2idx)
    print("Vocab size:", V)

    # we will also treat beginning of sentence and end of sentence as bigrams
    # START -> first word
    # last word -> END
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    # a matrix where:
    # row = last word
    # col = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(
        sentences, V, start_idx, end_idx, smoothing=0.1)

    # train a shallow neural network model
    D = 100
    W1 = np.random.randn(V, D) / np.sqrt(V)
    W2 = np.random.randn(D, V) / np.sqrt(D)

    losses = []
    epochs = 1
    lr = 1e-2

    def softmax(a):
        a = a - a.max()
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)

    # what is the loss if we set W = log(bigram_probs)?
    W_bigram = np.log(bigram_probs)
    bigram_losses = []

    t0 = datetime.now()
    for epoch in range(epochs):
        # shuffle sentences at each epoch
        random.shuffle(sentences)

        j = 0  # keep track of iterations
        for sentence in sentences:
            # do not one-hot encoded inputs and targets, remain as 1D list [CHANGED]
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            inputs = sentence[:n-1]
            targets = sentence[1:]

            # get output predictions, W1 is indexed with inputs instead of multiplying [CHANGED] save computation time
            hidden = np.tanh(W1[inputs])
            predictions = softmax(hidden.dot(W2))

            # keep track of the loss, double indexing trick on predictions [CHANGED]
            loss = - \
                np.sum(
                    np.log(predictions[np.arange(n - 1), targets])) / (n - 1)
            losses.append(loss)

            # do a gradient descent step [CHANGED]
            # do it after loss since the calculation of doutput will overwrite predictions
            # we don't want to make a copy because it would be slow
            doutput = predictions  # N x V
            doutput[np.arange(n - 1), targets] -= 1
            W2 = W2 - lr * hidden.T.dot(doutput)  # (D x N) (N x V)
            # (N x V) (V x D) * (N x D)
            dhidden = doutput.dot(W2.T) * (1 - hidden * hidden)
            # # for reference:
            # # original: W1 = W1 - lr * inputs.T.dot(dhidden) # VxN NxD --> VxD

            # fastest way
            np.subtract.at(W1, inputs, lr * dhidden)

            # test this
            # i = 0
            # for w in inputs: # don't include end token
            #   W1[w] = W1[w] - lr * dhidden[i]
            #   i += 1

            # vs this
            # oh_inputs = np.zeros((n - 1, V))
            # oh_inputs[np.arange(n - 1), sentence[:n-1]] = 1
            # W1 = W1 - lr * oh_inputs.T.dot(dhidden)

            # keep track of the bigram loss [CHANGED]
            # only do it for the first epoch to avoid redundancy
            if epoch == 0:
                bigram_predictions = softmax(W_bigram[inputs])
                bigram_loss = - \
                    np.sum(
                        np.log(bigram_predictions[np.arange(n - 1), targets])) / (n - 1)
                bigram_losses.append(bigram_loss)

            if j % 100 == 0:
                print("epoch:", epoch, "sentence: %s/%s" %
                      (j, len(sentences)), "loss:", loss)
            j += 1

    print("Elapsed time training:", datetime.now() - t0)
    plt.plot(losses)

    # plot a horizontal line for the bigram loss
    avg_bigram_loss = np.mean(bigram_losses)
    print("avg_bigram_loss:", avg_bigram_loss)
    plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')

    # plot smoothed losses to reduce variability

    def smoothed_loss(x, decay=0.99):
        y = np.zeros(len(x))
        last = 0
        for t in range(len(x)):
            z = decay * last + (1 - decay) * x[t]
            y[t] = z / (1 - decay ** (t + 1))
            last = z
        return y

    plt.plot(smoothed_loss(losses))
    plt.show()

    # plot W and bigram probs side-by-side
    # for the most common 200 words
    plt.subplot(1, 2, 1)
    plt.title("Neural Network Model")
    plt.imshow(np.tanh(W1).dot(W2))
    plt.subplot(1, 2, 2)
    plt.title("Bigram Probs")
    plt.imshow(W_bigram)
    plt.show()
