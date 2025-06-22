from vidyut.kosha import Kosha

kosha = Kosha("sans_tokenization/data/kosha")
for entry in kosha.get("gacCati"):
    print(entry)
