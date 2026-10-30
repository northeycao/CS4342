from collections import Counter
from datasets import load_dataset
import numpy as np

# Load dataset of children's stories
dataset = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=42)
counter = Counter()  # Keep track of most common words
ENOUGH_EXAMPLES = 100000
for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break
    # Ignore case and punctuation such as commas and quotation marks.
    words = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", " ").replace(".", " ").split(' ')
    filteredWords = [ w for w in words if w != "" ]
    counter.update(filteredWords)

# Select the most common words
NUM_WORDS = 501
topWords = [ w[0] for w in counter.most_common(NUM_WORDS-1) ]  # leave room for "."
topWords.append(".")  # end-of-sentence symbol "."
wordToIdxMap = { topWords[i]:i for i in range(NUM_WORDS) }  # map from a word to its index in the topWords list

# TASK 1 (training): estimate the three probability distributions P(x_1), P(x_2 | x_1), and P(x_{t+2} | x_t, x_{t+1}).
# TODO: initialize np.array's to represent the probability distributions
firstCounts = np.zeros(NUM_WORDS, dtype=np.float32)
secondCounts = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.float32)
nextCounts = np.zeros((NUM_WORDS, NUM_WORDS, NUM_WORDS), dtype=np.float32)
for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break
    # Split each story into sentences, ignoring case and punctuation.
    sentences = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", ".").split('. ')
    for sentence in sentences:
        # Convert each sentence into a word sequence
        sentence = sentence.replace(".", "")
        words = sentence.split(" ") + ["."]
        if not set(words).issubset(topWords):  # Ignore any sentence that contains ineligible words
            continue
        # TODO: process all the 3-grams in each sentence, but ignore any 3-gram if it contains 
        # a word that is not in topWords.
        # EXPLAIN - turning each word into its numbers
        wordInds = [ wordToIdxMap[w] for w in words]
        firstCounts[wordInds[0]] += 1
        secondCounts[wordInds[0], wordInds[1]] += 1

        for x in range(len(wordInds) - 2):
            firstInd = wordInds[x]
            secondInd = wordInds[x + 1]
            nextInd = wordInds[x + 2]
            nextCounts[firstInd, secondInd, nextInd] += 1


# TODO: normalize the probability distributions.
firstPr = firstCounts / firstCounts.sum()
secondTotal = secondCounts.sum(axis=1, keepdims=True)
secondPr = np.divide(secondCounts, secondTotal,
                     out=np.zeros_like(secondCounts),
                     where=secondTotal > 0)
nextTotal = nextCounts.sum(axis=2, keepdims=True)
np.divide(nextCounts, nextTotal,
          out=nextCounts,
          where=nextTotal > 0)
nextPr = nextCounts

# TASK 2 (testing/inference): use the probability distributions to generate 100 new "sentences".
# Note: given this relatively weak 3-gram model, not all sentences will be grammatical.
# This is ok for this assignment.
# To select from any probability distribution, you can use np.random.choice.
# TODO ...
endInd = wordToIdxMap["."]
maxLen = 50
numSen = 100

def sampleWordInd(probs):
    probs = probs.astype(np.float64)
    probs = probs / probs.sum()
    return np.random.choice(NUM_WORDS, p=probs)

def genSentence():
    firstInd = sampleWordInd(firstPr)
    genWords = [topWords[firstInd]]
    if firstInd == endInd:
        return genWords
    secondInd = sampleWordInd(secondPr[firstInd])
    genWords.append(topWords[secondInd])

    while secondInd != endInd and len(genWords) < maxLen:
        if nextPr[firstInd, secondInd].sum() == 0:
            break
        nextInd = sampleWordInd(nextPr[firstInd, secondInd])
        genWords.append(topWords[nextInd])
        firstInd, secondInd = secondInd, nextInd
    return genWords

np.random.seed(0)

madeSentences = []
for n in range(numSen):
    madeWords = genSentence()
    senText = " ".join(madeWords).replace(" .", ".")
    madeSentences.append(senText)
    print(f"{n + 1}. {senText}")

with open("homework1_autoregression_aparipally.txt", "w") as outputFile:
    for senText in madeSentences:
        outputFile.write(senText + "\n")

