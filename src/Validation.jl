function CorpusARI(state::State,model::Model,corpus::LdaCorpus)
  #for synthetic data, turn our mixed membership document vectors into max likelihood assignments
  # and check ARI between the ground truth and the state

  learned_max_clust = map(i->i[1], findmax(getTopicDist(state,model),dims=1)[2])
  true_max_clust = map(i->i[1], findmax(CorpusTopics(corpus),dims=1)[2])
  randindex(learned_max_clust,true_max_clust)
end

function DocsARI(state::State,corpus::LdaCorpus)
  #for synthetic data, find the topic ARI across [all terms in] all documents in the corpus
  learned_clust = AllTopics(state)
  true_clust = AllTopics(corpus)
  randindex(learned_clust,true_clust)
end
