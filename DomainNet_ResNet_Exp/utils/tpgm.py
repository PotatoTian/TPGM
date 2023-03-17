import copy

import torch
import torch.nn as nn

class TPGM(nn.Module):
    def __init__(self, model, norm_mode, exclude_list=[]) -> None:
        super().__init__()
        self.norm_mode = norm_mode
        self.exclude_list = exclude_list
        self.activation = nn.ReLU() 
        self.threshold = nn.Hardtanh(0, 1)
        self.constraints_name = []
        self.constraints = []
        self.create_contraint(model)
        self.constraints = nn.ParameterList(self.constraints)
        self.init = True
        

    def create_contraint(self, module):
        for name, para in module.named_parameters():
            if not para.requires_grad:
                continue
            if name not in self.exclude_list:
                self.constraints_name.append(name)
                temp = nn.Parameter(torch.Tensor([0]), requires_grad=True)
                self.constraints.append(temp)

    def apply_constraints(
        self,
        new,
        pre_trained,
        constraint_iterator,
        apply=False,
    ):
        for (name, new_para), anchor_para in zip(
            new.named_parameters(), pre_trained.parameters()
        ):
            if not new_para.requires_grad:
                continue
            if name not in self.exclude_list:
                alpha = self._project_ratio(
                    new_para,
                    anchor_para,
                    constraint_iterator,
                )
                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()
                if apply:
                    with torch.no_grad():
                        new_para.copy_(temp)
                else:
                    new_para.requires_grad = False
                    new_para.copy_(temp)

        self.init = False
     

    def _project_ratio(self, new, anchor, constraint_iterator):
        t = new.detach() - anchor.detach()

        if "l2" in self.norm_mode:
            norms = torch.norm(t)
        else:
            if new.dim() == 4:
                norms = torch.sum(torch.abs(t), dim=[1, 2, 3], keepdim=True).detach()
            elif new.dim() == 2:
                norms = torch.sum(torch.abs(t), dim=1, keepdim=True).detach()
            else:
                norms = torch.abs(t).detach()

        constraint = next(constraint_iterator)
        
        if self.init:
            with torch.no_grad():
                temp = norms.min()/2
                constraint.copy_(temp)

        self._clip(constraint, norms)
        ratio = self.threshold(self.activation(constraint) / (norms + 1e-8))
        return ratio

    def _clip(self, constraint, norms):
        if constraint <= 0:
            with torch.no_grad():
                constraint.copy_(torch.tensor(1e-8))

        if (self.activation(constraint) / (norms + 1e-8)).min() > 1:
            with torch.no_grad():
                constraint.copy_(norms.max())

    def forward(
        self,
        new=None,
        pre_trained=None,
        x=None,
        apply=False,
    ):
        constraint_iterator = iter(self.constraints)

        if apply:
            self.apply_constraints(
                new,
                pre_trained,
                constraint_iterator,
                apply=apply,
            )
        else:
            new_copy = copy.deepcopy(new)
            new_copy.eval()
            self.apply_constraints(new_copy, pre_trained, constraint_iterator)
            out = new_copy(x)
            return out


class tpgm_trainer(object):
    def __init__(
        self,
        model,
        pgmloader,
        norm_mode,
        proj_lr,
        max_iters,
        exclude_list = []
    ) -> None:
        self.device = torch.device("cuda")
        self.proj_lr = proj_lr
        self.max_iters = max_iters
        self.tpgm = TPGM(model, norm_mode=norm_mode, exclude_list=exclude_list).to(self.device)
        self.pre_trained = copy.deepcopy(model)
        self.pgm_optimizer = torch.optim.Adam(self.tpgm.parameters(), lr=self.proj_lr)
        self.pgmloader = pgmloader
        self.dataset_iterator = iter(self.pgmloader)
        self.criterion = torch.nn.CrossEntropyLoss()

    def tpgm_iters(self, model, apply=False):
        if not apply:
            self.count = 0
            while self.count < self.max_iters:
                try:
                    data = next(self.dataset_iterator)
                except StopIteration:
                    self.dataset_iterator = iter(self.pgmloader)
                    data = next(self.dataset_iterator)

                pgm_image, pgm_target = data
                pgm_image = pgm_image.to(self.device)
                pgm_target = pgm_target.to(self.device)

                outputs = self.tpgm(model,self.pre_trained, x=pgm_image)
                pgm_loss = self.criterion(outputs, pgm_target)
                self.pgm_optimizer.zero_grad() 
                pgm_loss.backward()
                self.pgm_optimizer.step()    
                self.count += 1

                if (self.count+1)%20 == 0:
                    print("{}/{} completed".format(self.count,self.max_iters))
        
        self.tpgm(model, self.pre_trained, apply=True)

