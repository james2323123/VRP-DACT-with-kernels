from torch.utils.data import Dataset
import torch
import pickle
import os

class FLP(object):

    NAME = 'flp'  # Facility Location Problem
    
    def __init__(self, p_size, init_val_met = 'random', with_assert = False, step_method = '2_opt', P = 10, DUMMY_RATE = 0):
        
        self.size = p_size
        self.do_assert = with_assert
        self.step_method = step_method
        self.init_val_met = init_val_met
        self.P = P
        print(f'FLP with {self.size} nodes.', ' Do assert:', with_assert)
        self.train()
    
    def eval(self, perturb = True):
        self.training = False
        self.do_perturb = perturb
        
    def train(self):
        self.training = True
        self.do_perturb = False
    
    def input_feature_encoding(self, batch):
        return batch['coordinates']
        
    def get_real_mask(self, visited_time):
        pass
    
    def get_initial_solutions(self, batch):
        
        batch_size = batch['coordinates'].size(0)
    
        def get_solution(methods):
            
            if methods == 'random':
                rec = torch.randint(0,2,(batch_size, self.size / 2))
                return rec
            
            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size).clone()
    
    def step(self, batch, rec, action, pre_bsf, solving_state = None, best_solution = None):

        bs = action.size(0)
        pre_bsf = pre_bsf.view(bs,-1)
        
        first = action[:,0].view(bs,1)
        second = action[:,1].view(bs,1)
        
        # TODO some step method for FLP
        if self.step_method  == 'toggle':
            next_state = self.toggle(rec, first, second)
        else:
            raise NotImplementedError()
        
        new_obj = self.get_costs(batch, next_state)
        
        now_bsf = torch.min(torch.cat((new_obj[:,None], pre_bsf[:,-1, None]),-1),-1)[0]
        
        reward = pre_bsf[:,-1] - now_bsf
        
        # update solving state
        solving_state[:,:1] = (1 - (reward > 0).view(-1,1).long()) * (solving_state[:,:1] + 1)
        
        if self.do_perturb:
            
            perturb_index = (solving_state[:,:1] >= self.P).view(-1)
            solving_state[:,:1][perturb_index.view(-1, 1)] *= 0
            pertrb_cnt = perturb_index.sum().item()
            
            if pertrb_cnt > 0:
                next_state[perturb_index] =  best_solution[perturb_index]

        return next_state, reward, torch.cat((new_obj[:,None], now_bsf[:,None]),-1), solving_state


    def toggle(self, solution, first, second, is_perturb = False):
        
        rec = solution.clone()
        
        # fix connection for first node
        argsort = solution.argsort()
        
        pre_first = argsort.gather(1,first)
        post_first = solution.gather(1,first)
        
        rec.scatter_(1,pre_first,post_first)
        
        # fix connection for second node
        post_second = rec.gather(1,second)
        
        rec.scatter_(1,second, first)
        rec.scatter_(1,first, post_second)
        
        return rec
        
    def check_feasibility(self, rec):
        # TODO if at least one facility is open
        pass
    
    
    def get_swap_mask(self, visited_time):
        
        bs, gs = visited_time.size()        
        selfmask = torch.eye(gs, device = visited_time.device).view(1,gs,gs)
        masks = selfmask.expand(bs,gs,gs).bool()
        
        return masks
   
    def get_costs(self, batch, rec):
        
        batch_size, size = rec.size()
        
        # check feasibility
        if self.do_assert:
            self.check_feasibility(rec)
        
        d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = batch['coordinates']
        length =  (d1  - d2).norm(p=2, dim=2).sum(1)
        
        return length
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return FLPDataset(*args, **kwargs)


class FLPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, DUMMY_RATE=None):
        
        super(FLPDataset, self).__init__()
        
        self.data = []
        self.size = size

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]

        else:
            self.data = [{
                'coordinates': torch.FloatTensor(self.size, 2).uniform_(0, 1),
                'cost_and_demand': torch.FloatTensor(self.size).uniform_(2, 5),
            } for i in range(num_samples)]
        
        self.N = len(self.data)
        
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        loc, cost_demand, *args = args

        return {
            'coordinates': torch.FloatTensor(loc),
            'cost_and_demand': torch.FloatTensor(cost_demand),
        }
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]
