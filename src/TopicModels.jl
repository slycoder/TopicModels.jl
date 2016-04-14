module TopicModels

import Base.length

typealias RaggedMatrix{T} Array{Array{T,1},1}

type Corpus
  documents::RaggedMatrix{Int64}
  weights::RaggedMatrix{Float64}

  Corpus(documents::RaggedMatrix{Int64},
         weights::RaggedMatrix{Float64}) = begin
    return new(
      documents,
      weights
    )
  end
  
  Corpus(documents::RaggedMatrix{Int64}) = begin
    weights = map(documents) do doc
      ones(Float64, length(doc))
    end
    return new(
      documents,
      weights
    )
  end
end

type Model
  alphaPrior::Vector{Float64}
  betaPrior::Float64
  topics::Array{Float64,2}
  topicSums::Vector{Float64}
  documentSums::Array{Float64,2}
  assignments::RaggedMatrix{Int64}
  frozen::Bool
  corpus::Corpus

  Model(alphaPrior::Vector{Float64}, 
        betaPrior::Float64, 
        V::Int64, 
        corpus::Corpus) = begin
    K = length(alphaPrior)
    m = new(
      alphaPrior,
      betaPrior,
      zeros(Float64, K, V), # topics
      zeros(Float64, K), # topicSums
      zeros(Float64, K, length(corpus.documents)), #documentSums
      fill(Array(Int64, 0), length(corpus.documents)), # assignments
      false,
      corpus
    )
    initializeAssignments(m)
    return m
  end

  Model(trainedModel::Model, corpus::Corpus) = begin
    m = new(
      trainedModel.alphaPrior,
      trainedModel.betaPrior,
      trainedModel.topics,
      trainedModel.topicSums,
      trainedModel.documentSums,
      fill(Array(Int64, 0), length(corpus.documents)),
      true,
      corpus
    )
    initializeAssignments(m)
    return m
  end
end

function length(corpus::Corpus)
  return length(corpus.documents)
end

function initializeAssignments(model::Model)
  for dd in 1:length(model.corpus)
    @inbounds words = model.corpus.documents[dd]
    @inbounds model.assignments[dd] = fill(0, length(words))
    for ww in 1:length(words)
      @inbounds word = words[ww]
      topic = sampleMultinomial(model.alphaPrior)
      @inbounds model.assignments[dd][ww] = topic
      updateSufficientStatistics(
        word, topic, dd, model.corpus.weights[dd][ww], model)
    end
  end
  return
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
                          model::Model,
                          out::Vector{Float64})
  V = size(model.topics, 2)
  for ii in 1:length(out)
    u = (model.documentSums[ii, document] + model.alphaPrior[ii]) * 
        (model.topics[ii, word] + model.betaPrior) / 
        (model.topicSums[ii] + V * model.betaPrior)
    @inbounds out[ii] = u
  end
  return
end

function sampleWord(word::Int,
                    document::Int,
                    model::Model,
                    p::Vector{Float64})
  wordDistribution(word, document, model, p)
  sampleMultinomial(p)
end


function updateSufficientStatistics(word::Int64, 
                                    topic::Int64,
                                    document::Int64,
                                    scale::Float64, 
                                    model::Model)
  fr = Float64(!model.frozen)
  @inbounds model.documentSums[topic, document] += scale
  @inbounds model.topicSums[topic] += scale * fr
  @inbounds model.topics[topic, word] += scale * fr
  return
end

function sampleDocument(document::Int,
                        model::Model)
  @inbounds words = model.corpus.documents[document]
  Nw = length(words)
  @inbounds weights = model.corpus.weights[document]
  K = length(model.alphaPrior)
  p = Array(Float64, K)
  @inbounds assignments = model.assignments[document]
  for ii in 1:Nw
    @inbounds word = words[ii]
    @inbounds oldTopic = assignments[ii]
    updateSufficientStatistics(word, oldTopic, document, -weights[ii], model)
    newTopic = sampleWord(word, document, model, p)
    @inbounds assignments[ii] = newTopic
    updateSufficientStatistics(word, newTopic, document, weights[ii], model)
  end
  return
end

function sampleCorpus(model::Model)
  for ii in 1:length(model.corpus)
    sampleDocument(ii, model)
  end
  return
end

# Note, files are zero indexed, but we are 1-indexed.
function termToWordSequence(term::AbstractString)
  parts = split(term, ":")
  fill(parse(Int64, parts[1]) + 1, parse(Int64, parts[2]))
end 

# The functions below are designed for public consumption
function trainModel(model::Model, 
                    numIterations::Int64)
  for ii in 1:numIterations
    println(string("Iteration ", ii, "..."))
    sampleCorpus(model)
  end
  return
end

function topTopicWords(model::Model,
                       lexicon::Array{ASCIIString,1},
                       numWords::Int64)
  [lexicon[reverse(sortperm(model.topics'[1:end, row]))[1:numWords]]
   for row in 1:size(model.topics,1)]
end

function readDocuments(stream)
    lines = readlines(stream)
    convert(RaggedMatrix{Int64},
            [vcat([termToWordSequence(term) for term in split(line, " ")[2:end]]...)
             for line in lines])
end

function readLexicon(stream)
  lines = readlines(stream)
  map(chomp, convert(Array{AbstractString,1}, lines))
end

export Corpus,
       Model,
       readDocuments,
       readLexicon,
       topTopicWords,
       trainModel
end
