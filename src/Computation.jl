struct Model
  alphaPrior::Vector{Float64} # concentration parameter for the symmetric Dirichlet prior on document topics
  betaPrior::Vector{Float64} # concentration parameter for the symmetric Dirichlet prior on words
  corpus::AbstractCorpus

  # initialize an untrained model
  Model(alphaPrior::Vector{Float64},
        betaPrior::Vector{Float64},
        corpus::AbstractCorpus) = begin
    K = length(alphaPrior)
    m = new(
      alphaPrior,
      betaPrior,
      corpus)
    return m
  end

  # initialize a trained model
  Model(trainedModel::Model,
        corpus::AbstractCorpus) = begin
    m = new(
      trainedModel.alphaPrior,
      trainedModel.betaPrior,
      corpus
    )
    return m
  end
end

struct State
  topics::Array{Float64,2}
  topicSums::Vector{Float64}
  docSums::Array{Float64,2}
  assignments::Array{Array{Int64,1},1}
  conditionals::Array{Array{Float64,2},1} # the p paramter for the word assignment (cat/multinom) variable
  frozen::Bool

  # randomly initialize the state
  State(model::Model,
        corpus::AbstractCorpus) = begin # length of the lexicon
    K = length(model.alphaPrior)
    s = new(
      zeros(Float64, K, length(corpus.lexicon)), # topics
      zeros(Float64, K), # topicSums
      zeros(Float64, K, length(corpus.docs)), #docSums
      fill(Array{Int64,1}(undef,0), length(corpus.docs)), # assignments
      fill(Array{Int64,2}(undef,0,K), length(corpus.docs)),
      false
    )
    initializeAssignments(model,s,corpus)
    return s
  end

  # initialize the state from a trained model
  State(topics::Array{Float64,2},
        topicSums::Vector{Float64},
        docSums::Array{Float64,2},
        assignments::Array{Array{Int64,1},1},
        conditionals::Array{Array{Float64,2},1},
        frozen::Bool) = begin # length of the lexicon
    s = new(
      topics,
      topicSums,
      docSums,
      assignmens,
      conditionals,
      frozen
    )
    return s
  end
end

function AllTopics(state::State)
  alltopics = []
  for i in 1:length(state.assignments)
    append!(alltopics,state.assignments[i])
  end
  return convert(Array{Int,1},alltopics)
end

function initializeAssignments(model::Model,state::State,corpus::AbstractCorpus)
  for dd in 1:length(corpus)
    @inbounds words = corpus.docs[dd].terms
    @inbounds state.assignments[dd] = zeros(length(words))
    @inbounds state.conditionals[dd] = zeros(length(words), length(model.alphaPrior))
    for ww in 1:length(words)
      @inbounds word = words[ww]
      @inbounds state.conditionals[dd][ww,:] = model.alphaPrior
      topic = sampleMultinomial(ww,dd,state)
      @inbounds state.assignments[dd][ww] = topic
      updateSufficientStatistics(word, topic, dd,
                                  model.corpus.weights[dd][ww],
                                  state)
    end
  end
  return
end


function sampleMultinomial(word_ind::Int64,
                           document::Int64,
                           state::State)
  cond = state.conditionals[document][word_ind,:]
  pSum = sum(cond)
  r = rand() * pSum
  K = length(cond)
  for k in 1:K
    if r < cond[k]
      return k
    else
      @inbounds r -= cond[k]
    end
  end
  return 0
end

function cond_word(word::Int,
                   word_ind::Int,
                   document::Int,
                   model::Model,
                   state::State)
  V = size(state.topics, 2)
  for ii in 1:length(model.alphaPrior)
    @inbounds state.conditionals[document][word_ind,ii] =
        (state.docSums[ii, document] + model.alphaPrior[ii]) *
        (state.topics[ii, word] + model.betaPrior[word]) /
        (state.topicSums[ii] + V * model.betaPrior[word])
  end
  return
end

function log_beta(x::Vector{Float64})
  # compute natural log of the multivariate beta function
  lb = sum(loggamma.(x))
  lb -= loggamma(sum(x))
end

function joint_log_p(model::Model,
                     state::State)
  #calculate the full joint log likelihood, this is usefull for testing
  log_pz = 0
  for k in 1:length(model.alphaPrior)
    @inbounds log_pz += (log_beta(state.topics[k,:] .+ model.betaPrior) -
                log_beta(model.betaPrior))
  end
  for d in 1:length(model.corpus)
    @inbounds log_pz += (log_beta(state.docSums[:,d] .+ model.alphaPrior) -
                log_beta(model.alphaPrior))
  end
  return log_pz
end

function sampleWord(word::Int,
                    word_ind::Int,
                    document::Int,
                    model::Model,
                    state::State)
  cond_word(word, word_ind, document, model, state)
  sampleMultinomial(word_ind, document, state)
end


function updateSufficientStatistics(word::Int64,
                                    topic::Int64,
                                    document::Int64,
                                    scale::Float64,
                                    state::State)
  fr = Float64(!state.frozen)
  @inbounds state.docSums[topic, document] += scale
  @inbounds state.topicSums[topic] += scale * fr
  @inbounds state.topics[topic, word] += scale * fr
  return
end

@doc raw"""
    getTermDist(state::State, model::Model)

Compute ``\phi_{k,v} = \frac{\Psi_{k,v} + \beta_t}{\left( \sum^V_{v'=1} \Psi_{k,v'} + \beta_{v'}\right)}``

Where ``\vec{ϕ_v}`` parameterizes the V-dimensional categorical distribution of a word.

Updates the `termDist` attribute of `state`
"""
function getTermDist(state::State, model::Model)
  Phi = Array{Float64,2}(undef,length(model.alphaPrior),length(model.betaPrior))
  for topic in 1:length(model.alphaPrior)
    Phi[topic,:] = (state.topics[topic,:] .+ model.betaPrior) ./ (state.topicSums[topic] + sum(model.betaPrior))
  end
  return Phi
end

@doc raw"""
    getTopicDist(state::State, model::Model)

Compute ``\theta_{k,m} = \frac{\Omega_{k,m} + \alpha_k}{\left( \sum^K_{k'=1} \Omega_{k,m'} + \alpha_{k'}\right)}``

Where ``\vec{\theta_m}`` parameterizes the K-dimensional categorical distribution of a document.

Updates the `topicDist` attribute of `state`
"""
function getTopicDist(state::State, model::Model)
  Theta = Array{Float64,2}(undef,length(model.alphaPrior),length(model.corpus))
  for doc in 1:length(model.corpus)
    Theta[:,doc] = (state.docSums[:,doc] .+ model.alphaPrior) ./ (sum(state.docSums[:,doc]) + sum(model.alphaPrior))
  end
  return Theta
end

function sampleDocument(document::Int,
                        model::Model,
                        state::State)
  words = model.corpus.docs[document].terms
  Nw = length(words)
  @inbounds weights = model.corpus.weights[document]
  K = length(model.alphaPrior)
  @inbounds assignments = state.assignments[document]
  for ii in 1:Nw
    word = words[ii]
    oldTopic = assignments[ii]
    updateSufficientStatistics(word, oldTopic, document, -weights[ii], state)
    newTopic = sampleWord(word, ii, document, model, state)
    @inbounds assignments[ii] = newTopic
    updateSufficientStatistics(word, newTopic, document, weights[ii], state)
  end
  return
end

function sampleCorpus(model::Model, state::State)
  for ii in 1:length(model.corpus)
    sampleDocument(ii, model, state)
  end
  return
end

# The functions below are designed for public consumption
function trainModel(model::Model,
                    state::State,
                    numIterations::Int64)
  for ii in 1:numIterations
    println(string("Iteration ", ii, "..."))
    sampleCorpus(model, state)
  end
  return
end

function topTopicWords(model::Model,
                       state::State,
                       numWords::Int64)
  [model.corpus.lexicon[reverse(sortperm(state.topics'[1:end, row]))[1:numWords]]
   for row in 1:size(state.topics,1)]
end
