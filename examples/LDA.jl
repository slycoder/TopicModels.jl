using TopicModels, Plots, UMAP

exdir = joinpath(dirname(pathof(TopicModels)), "..", "examples")

testDocuments = readDocs(open(joinpath(exdir, "cora.documents")))
testLexicon = readLexicon(open(joinpath(exdir, "cora.lexicon")))

corpus = Corpus(testDocuments,testLexicon)
model = Model(fill(0.1, 10), fill(0.01,length(testLexicon)), corpus)
state = State(model,corpus)

#@time Juno.@run trainModel(model, state, 30)
@time trainModel(model, state, 30)
topWords = topTopicWords(model, state, 10)

embedding = umap(state.topics, 2, n_neighbors=10)
maxlabels = vec(map(i->i[1], findmax(state.topics,dims=1)[2]))
scatter(embedding[1,:], embedding[2,:], zcolor=maxlabels, title="UMAP: Max on Learned", marker=(2, 2, :auto, stroke(0)))


k = 10
lexLength = 1000
corpLambda = 1000 # poisson parameter for random doc length
corpLength = 100
scaleK = 0.01
scaleL = 0.01
testCorpus = LdaCorpus(k, lexLength, corpLambda, corpLength, scaleK, scaleL)

testModel = Model(testCorpus.alpha, testCorpus.beta, testCorpus)
testState = State(testModel, testCorpus)
@time trainModel(testModel, testState, 100)

# compute validation metrics on a single fit
CorpusARI(testState,testModel,testCorpus)
DocsARI(testState,testCorpus)

# visualize the fit
@time embedding = umap(testState.topics, 2;n_neighbors=10)

maxlabels = vec(map(i->i[1], findmax(CorpusTopics(testCorpus),dims=1)[2]))
scatter(embedding[1,:], embedding[2,:], zcolor=maxlabels, title="UMAP: True on Learned", marker=(2, 2, :auto, stroke(0)))
