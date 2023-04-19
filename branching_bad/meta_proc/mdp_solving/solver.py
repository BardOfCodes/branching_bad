

class BootstrappedLearning:
    
    def __init__(self, config):
        ...
        
    
    def solve_mdp(self, model, train_dataset, val_dataset, dsl_executor):
        
        converged = False
        while(not converged):
            # fine-tune the model on the dataset
            model = self.fine_tune(model, train_dataset, val_dataset)
            # update dataset:
            train_dataset = self.update_dataset(model, train_dataset, dsl_executor)
            
            converged = self.check_convergence(val_dataset, model, dsl_executor)
            

        # if converged, create the model, train_dataset for abstraction discovery.
        updated_dataset = self.update_dataset(model, train_dataset, dsl_executor, final=True)
        
        return model, updated_dataset

        
    def fine_tune(self, model, train_dataset, val_dataset):
        
        # Inner loop
        plad_converged = False
        while(not plad_converged):
            
            self.mle_training(model, train_dataset)
            
            val_performance = self.test_on_val(model, val_dataset)
            
            plad_converged = self.check_plad_convergence(val_performance)
        
        return model