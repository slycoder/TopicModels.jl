### Document
abstract type AbstractDocument end

struct LdaDocument <: AbstractDocument
  # this is a fully observed data from the LDA model
  theta::Array{Float64,1} # the topic probs for the doc
  z::Array{Int64,1} # the topic for each word
  terms::Array{Int64,1} # the word tokens

  LdaDocument(alpha::Array{Float64,1},
                  P::Array{Float64,2},
                  N::Int64) = begin # length of the doc
    d = new(
      Array{Float64,1}(undef,size(P,2)),
      Array{Int64,1}(undef,N),
      Array{Int64,1}(undef,N)
    )
    GenerateDoc(d,alpha,P)
    return d
  end

  LdaDocument(theta::Array{Float64,1},
              z::Array{Int64,1},
              terms::Array{Int64,1}) = begin
    d = new(theta,z,N)
    return d
  end
end

function GenerateDoc(doc::LdaDocument,
                     alpha::Array{Float64,1},
                     Phi::Array{Float64,2})
    dd = Dirichlet(alpha)
    doc.theta .= vec(rand(dd,1))
    cat = Categorical(vec(doc.theta))
    doc.z .= rand(cat,length(doc))
    for i in 1:length(doc)
        @inbounds dc = Categorical(Phi[:,doc.z[i]])
        @inbounds doc.terms[i] = rand(dc,1)[1]
    end
    return
end

struct Document <: AbstractDocument
  #this is actual data, where only the terms are observed
  terms::Array{Int64,1} # the word tokens
  Document(terms::Array{Int64,1}) = new(terms)
end

function length(doc::AbstractDocument)
  return size(doc.terms,1)
end

### Corpus
abstract type AbstractCorpus end

struct LdaCorpus <: AbstractCorpus
  # this is a fully observed data from the LDA model
  docs::Array{LdaDocument,1}
  alpha::Array{Float64,1}
  beta::Array{Float64,1}
  Phi::Array{Float64,2}
  weights::Array{Array{Float64,1}} # only unweighted terms supported
  lexicon::Array{String,1}

  LdaCorpus(k::Int64,
            lexLength::Int64,
            corpLambda::Int64,
            corpLength::Int64,
            scaleK::Float64,
            scaleL::Float64) = begin # length of the doc
    w = Array{Array{Float64,1},1}(undef,corpLength)
    lex = string.([1:1:lexLength;]) # there is no
    a = fill(scaleK,k) # scale parameter for the Dirichlet topic prior
    b = fill(scaleL,lexLength) # scale parameter for the Dirichlet token prior
    dl = Poisson(corpLambda)
    docLengths = rand(dl,corpLength) # the lengths of the docs in the corpus
    db = Dirichlet(b)
    P = rand(db,k) # the Dirichlet token prior, containing one lexLength vector for each k
    d = Array{LdaDocument,1}(undef,corpLength)
    for i in 1:corpLength
      w[i] = ones(docLengths[i])
      @inbounds d[i] = LdaDocument(a,P,docLengths[i])
    end
    return new(d, a, b, P, w, lex)
  end

  LdaCorpus(docs::Array{LdaDocument,1}, # the documents
            alpha::Array{Float64,1},
            beta::Array{Float64,1},
            Phi::Array{Float64,2},
            weights::Array{Float64,1},
            lexicon::Array{String,1}) = begin
    c = new(docs,alpha,beta,Phi,weights)
    return c
  end
end

function CorpusTopics(corpus::LdaCorpus)
  cat(dims=2,map(i->vec(i.theta), corpus.docs)...) # get a 2d array of (document wise) mixed membership for the corpus
end

function AllTopics(corpus::LdaCorpus)
  alltopics = []
  for i in 1:length(corpus)
    append!(alltopics,corpus.docs[i].z)
  end
  return convert(Array{Int,1},alltopics)
end

struct Corpus <: AbstractCorpus
  docs::Array{Document,1}
  weights::Array{Array{Float64,1},1}
  lexicon::Array{String,1}

  Corpus(docs::Array{Document,1},
         weights::Array{Array{Float64,1},1},
         lexicon::Array{String,1}) = begin
    return new(
      docs,
      weights,
      lexicon
    )
  end

  Corpus(docs::Array{Document,1},
         lexicon::Array{String,1}) = begin
    return new(
      docs,
      map(x -> ones(Float64,length(x)), docs), # no weights
      lexicon
    )
  end
end

function length(corpus::AbstractCorpus)
  return length(corpus.docs)
end

# Expand  a term:count pair into a <count>-length sequence [term, term, ....]
function termToWordSequence(term::AbstractString)
  parts = split(term, ":")
  fill(parse(Int64, parts[1]) + 1, parse(Int64, parts[2]))
end

function readDocs(stream)
    corpus = readlines(stream)
    docs = Array{Document,1}(undef,length(corpus))
    for i in 1:length(corpus)
      @inbounds terms = split(corpus[i], " ")[2:end]
      @inbounds docs[i] = Document(termToWordSequence(terms[1]))
      for ii in 2:length(terms)
        @inbounds append!(docs[i].terms, termToWordSequence(terms[ii]))
      end
    end
    return docs
end

function readLexicon(stream)
  lines = readlines(stream)
  chomped = map(chomp, convert(Array{AbstractString,1}, lines))
  convert(Array{String,1},chomped) # convert from substrings
end
