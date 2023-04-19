
class AbstractionDiscovery:
    
    def __init__(self, config):
        ...
    
    def discover_abstractions(self, dataset, dsl_executor):
        
        # Do AD on "clean" dataset
        clean_dataset = self.prune_dataset(dataset)
        
        # retrieve candidates:
        candidate_abstractions = self.retrieve_candidates(clean_dataset, dsl_executor)
        
        # perform dataset rewrites with each and score?
        
        # select tok-k abstractions.
        
        