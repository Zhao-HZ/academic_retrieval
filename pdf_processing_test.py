from wordsegment import load, segment

# load()
# texts = ["iloveyou", "whatisyourname", "thankyouverymuch"]
# for text in texts:
#     print(f"{text} -> {' '.join(segment(text))}")

load()
print(segment("Iloveyou"))
