module TopicModels

typealias RaggedMatrix{T} Array{Array{Int64,1},1}
typealias Corpus RaggedMatrix{Int64}

type Model
  alphaPrior::Array{Float64,1}
  betaPrior::Float64
  topics::Array{Int64,2}
  topicSums::Array{Int64,1}
  documentSums::Array{Int64,2}
  assignments::RaggedMatrix{Int64}

  Model(alphaPrior::Array{Float64,1}, 
        betaPrior::Float64, 
        V::Int64, 
        corpus::Corpus) = begin
    K = length(alphaPrior)
    m = new(
      alphaPrior,
      betaPrior,
      zeros(Int64, K, V), # topics
      zeros(Int64, K), # topicSums
      zeros(Int64, K, length(corpus)), #documentSums
      fill(Array(Int64, 0), length(corpus)) # assignments
    )
    for dd in 1:length(corpus)
      m.assignments[dd] = fill(0, length(corpus[dd])) 
      for ww in 1:length(corpus[dd])
        word = corpus[dd][ww]
        topic = sampleMultinomial(alphaPrior)
        m.assignments[dd][ww] = topic
        updateSufficientStatistics(word, topic, dd, 1, m)
      end
    end
    return m
  end
end

function sampleMultinomial(p::Array{Float64,1})
  pSum = sum(p)
  r = rand() * pSum
  K = length(p)
  for k in 1:K
    if r < p[k]
      return k
    else
      r -= p[k]
    end
  end
  return 0
end

function wordDistribution(word::Int,
                          document::Int,
                          model::Model)
  V = size(model.topics, 2)
  (model.documentSums[1:end,document] + model.alphaPrior) .* 
    (model.topics[1:end, word] + model.betaPrior) ./ 
    (model.topicSums + V * model.betaPrior)
end

function sampleWord(word::Int,
                    document::Int,
                    model::Model)
  p = wordDistribution(word, document, model)
  sampleMultinomial(p)
end


function updateSufficientStatistics(word::Int, 
                                    topic::Int,
                                    document::Int,
                                    scale::Int, 
                                    model::Model)
  model.topics[topic, word] += scale
  model.topicSums[topic] += scale
  model.documentSums[topic, document] += scale
end

function sampleDocument(words::Array{Int64,1},
                        document::Int,
                        model::Model) 
  Nw = length(words)
  for ii in 1:Nw
    word = words[ii]
    oldTopic = model.assignments[document][ii] 
    updateSufficientStatistics(word, oldTopic, document, -1, model)
    newTopic = sampleWord(word, document, model)
    model.assignments[document][ii] = newTopic
    updateSufficientStatistics(word, newTopic, document, 1, model)
  end
end

function sampleCorpus(corpus::Corpus,
                      model::Model)
  for ii in 1:length(corpus)
    sampleDocument(corpus[ii], ii, model)
  end
end

# Note, files are zero indexed, but we are 1-indexed.
function termToWordSequence(term::String)
  parts = split(term, ":")
  fill(int64(parts[1]) + 1, int64(parts[2]))
end 

# The functions below are designed for public consumption
function trainModel(corpus::Corpus,
                    model::Model, 
                    numIterations::Int64)
  for ii in 1:numIterations
    println(string("Iteration ", ii, "..."))
    sampleCorpus(corpus, model)
  end
end

function topTopicWords(model::Model,
                       lexicon::Array{ASCIIString,1},
                       numWords::Int64)
  [lexicon[reverse(sortperm(model.topics'[1:end, row]))[1:numWords]]
   for row in 1:size(model.topics,1)]
end

function readDocuments(stream)
  lines = readlines(stream)
  convert(
    RaggedMatrix{Int64},
    [apply(vcat, [termToWordSequence(term) for term in split(line, " ")[2:end]])
     for line in lines])
end

function readLexicon(stream)
  lines = readlines(stream)
  map(chomp, convert(Array{String,1}, lines))
end

# Test stuff
testDocuments = readDocuments(open("cora.documents"))
testLexicon = readLexicon(open("cora.lexicon"))
model = Model(fill(0.1, 10), 0.01, length(testLexicon), testDocuments)
trainModel(testDocuments, model, 30)

topWords = topTopicWords(model, testLexicon, 21)

end
