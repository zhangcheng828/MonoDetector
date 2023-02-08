import torch
import torch.nn as nn
import torch.nn.functional as F


class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'loss_center_heatmap':[],
                           'wh_loss':[], 
                           'offset_2d_loss':[],
                           'offset_3d_loss':['wh_loss','offset_2d_loss'], 
                           'loss_dim':['wh_loss','offset_2d_loss'], 
                           'loss_heading':['wh_loss','offset_2d_loss'], 
                           'loss_depth':['wh_loss','loss_dim','offset_2d_loss']}

    def compute_weight(self,current_loss, epoch):
        T=200
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)), 1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]] 
                    print(control_weight)   
                    loss_weights[current_topic] = time_value**(1-control_weight)
            
            #pop first list
            self.past_losses.pop(0)

        self.past_losses.append(eval_loss_input)   
        return loss_weights
