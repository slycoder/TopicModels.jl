using TopicModels

exdir = Pkg.dir("TopicModels", "examples")

testDocuments = readDocuments(open(joinpath(exdir, "cora.documents")))
testLexicon = readLexicon(open(joinpath(exdir, "cora.lexicon")))

corpus = Corpus(testDocuments)

model = Model(fill(0.1, 10), 0.01, length(testLexicon), corpus)

@time trainModel(model, 30)

topWords = topTopicWords(model, testLexicon, 21)
