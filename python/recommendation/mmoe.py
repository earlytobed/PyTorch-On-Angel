import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, experts_in, experts_out, experts_hidden):
        super(Expert, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(experts_in, experts_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(experts_hidden, experts_out),
        )

    def forward(self, x):
        return self.expert(x)


class Tower(nn.Module):
    def __init__(self, towers_in, towers_out, towers_hidden):
        super(Tower, self).__init__()
        self.tower = nn.Sequential(
            nn.Linear(towers_in, towers_hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(towers_hidden, towers_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.tower(x)


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Sigmoid(dim=1),
        )

    def forward(self, x):
        return self.tower(x)


class MMoE(nn.Module):
    def __init__(self, input_dim=-1, n_fields=-1, experts_out=-1, experts_hidden=-1, towers_hidden=-1, num_experts=6, tasks=1):
        super(MMoE, self).__init__()
        # begin something
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.BCELoss(reduction="mean")
        # end
        self.input_dim = input_dim
        self.n_fields = n_fields
        # self.embedding_dim = embedding_dim

        self.experts_out = experts_out

        self.experts = nn.ModuleList([Expert(input_dim, experts_out, experts_hidden) for i in range(num_experts)])
        self.w_gates = [nn.Parameter(torch.randn(input_dim, num_experts), requires_grad=True) for i in range(tasks)]

        self.towers = nn.ModuleList([Tower(experts_out, 1, towers_hidden) for i in range(tasks)])

        """兼容"""
        self.embedding_dim = 10
        self.mats = self.w_gates
        """兼容"""

    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats, fields=torch.Tensor([])):
        # experts_o_tensor = torch.stack(list(map(lambda f: f(x), self.experts)))
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(x @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        # final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return self.towers[0](tower_input[0])

    def forward(self, batch_size, index, feats, values):
        # batch: coo
        # index = torch.from_numpy(batch.row).to(torch.long)
        # feats = torch.from_numpy(batch.col).to(torch.long)
        # values = torch.from_numpy(batch.data)
        return self.forward_()

    @torch.jit.export
    def loss(self, pred, gt):
        task1_gt = gt.view(-1, 1)
        task1_pred = pred
        return self.loss_fn(task1_pred, task1_gt)

    @torch.jit.export
    def get_type(self):
        return "BIAS_WEIGHT_EMBEDDING_MATS"

    @torch.jit.export
    def get_name(self):
        return "MMoE"


if __name__ == "__main__":
    model = MMoE(input_dim=148, num_experts=6, experts_out=16, experts_hidden=32, towers_hidden=8, tasks=1)
    script_module = torch.jit.script(model)
    script_module.save("MMoE.pt")
    print(model)
