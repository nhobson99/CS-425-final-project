import re as regex

def gunning_fog(line):
    words = "".join(line.strip().split("/")).split()
    num_words = len(words)
    num_sent = len(line.split('.'))

    long_words = 0
    for word in words:
        letters = regex.compile('[a-zA-Z]')
        if (letters.search(word) == None):
            num_words -= 1
        syllables = regex.compile("[aeiou]+")
        num_syllables = len(syllables.findall(word)) # this won't be exact, but it's close enough
        if (word.endswith("ed") or word.endswith("es") or word.endswith("ing")):
            num_syllables -= 1
        if (num_syllables >= 3):
            long_words += 1
 
    if (num_words == 0):
        return 0

    ASL = num_words / num_sent
    PHW = 100 * long_words / num_words
 
    return 0.4 * (ASL + PHW)