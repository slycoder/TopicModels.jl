using Test, TopicModels, Random
using TopicModels: updateSufficientStatistics, joint_log_p #non-exported fns we need


# use the equality of likelihood ratio to test that the conditional distribution is consistent with the joint distribution
@testset "LDA docs" begin
    # generate some data from LDA where the doclength is Poisson
    k = 7
    lexLength = 10
    corpLambda = 10 # poisson parameter for random doc length
    corpLength = 10
    scaleK = 0.1
    scaleL = 0.1
    Random.seed!(123)

    corpus = LdaCorpus(k, lexLength, corpLambda, corpLength, scaleK, scaleL)

    model = Model(corpus.alpha, corpus.beta, corpus)
    state = State(model, corpus)
    trainModel(model, state, 10) # update all the state variables

    # pick a random doc/word to iterate the sampler
    doc_ind = rand(1:corpLength)
    word_ind = rand(1:length(corpus.docs[doc_ind]))
    word = corpus.docs[doc_ind].terms[word_ind]

    conditional = state.conditionals[doc_ind][word_ind,:]
    oldTopic = copy(state.assignments[doc_ind][word_ind])  # the original word token

    newTopic = rand(collect(1:k)[1:end .!= oldTopic],1) # a different word token
    newTopic = Int64(newTopic[1])

    #get the original state probs
    joint_Lw = copy(joint_log_p(model,state)) # log prob of the full joint under original topic for <doc><word>
    cond_Lw = log(state.conditionals[doc_ind][word_ind,oldTopic]/sum(state.conditionals[doc_ind][word_ind,:])) # log conditional p(z=k|...)
    cond_Lw_new = log(state.conditionals[doc_ind][word_ind,newTopic]/sum(state.conditionals[doc_ind][word_ind,:])) # log conditional p(z=k|...)

    updateSufficientStatistics(word, oldTopic, doc_ind, -model.corpus.weights[doc_ind][word_ind], state) #remove counts for the old topic
    updateSufficientStatistics(word, newTopic, doc_ind, model.corpus.weights[doc_ind][word_ind], state) #update stats for new topic
    joint_Lw_new = copy(joint_log_p(model,state)) # log prob of the full joint under original topic for <doc><word>

    print("joint_Lw: ", joint_Lw, "\n")
    print("cond_Lw: ", cond_Lw, "\n")

    print("joint_Lw_new: ", joint_Lw_new, "\n")
    print("cond_Lw_new: ", cond_Lw_new, "\n")

    print("joint_LR: ", joint_Lw_new-joint_Lw, "\n")
    print("cond_LR: ", cond_Lw_new-cond_Lw, "\n")
    print("old Topic: ", oldTopic, "\n")
    print("new Topic: ", newTopic, "\n")

    @test isless(abs(joint_Lw_new-joint_Lw - cond_Lw_new+cond_Lw),1e-5)
end
