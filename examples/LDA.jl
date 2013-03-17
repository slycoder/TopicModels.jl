load("TopicModels")
using TopicModels

testDocuments = readDocuments(open("cora.documents"))
testLexicon = readLexicon(open("cora.lexicon"))
model = Model(fill(0.1, 10), 0.01, length(testLexicon), testDocuments)
trainModel(testDocuments, model, 30)
topWords = topTopicWords(model, testLexicon, 21)
