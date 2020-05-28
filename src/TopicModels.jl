module TopicModels

#Imports
import Base.length

using Random, Distributions, Plots, UMAP
using SpecialFunctions: loggamma
using Clustering: randindex

#Exports
export Corpus,
       LdaCorpus,
       Model,
       State,
       readDocs,
       readLexicon,
       termToWordSequence,
       topTopicWords,
       trainModel,
       CorpusTopics,
       CorpusARI,
       DocsARI

#Data that we make or find in real life:
include("Data.jl")

#Bayesian learning and inference:
include("Computation.jl")

#Stuff like perplexity and ARI:
include("Validation.jl")
end #module