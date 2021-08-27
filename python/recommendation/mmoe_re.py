import torch
from torch import nn
from typing import List


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


class MMoE(nn.Module):
    def __init__(self, input_dim=-1, n_fields=-1, embedding_dim=-1, experts_hidden=-1, experts_out=-1, towers_hidden=-1, towers_out=1, num_experts=6, tasks=1):
        super(MMoE, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.BCELoss(reduction="mean")

        self.input_dim = input_dim
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim

        self.experts_out = experts_out
        self.num_experts = num_experts
        self.tasks = tasks

        """Angel Params"""
        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))
        # weights
        self.weights = torch.nn.Parameter(torch.zeros(1, 1))
        # embeddings
        self.embeddings = torch.nn.Parameter(torch.zeros(embedding_dim))

        """mats"""
        self.mats = []
        # experts
        for i in range(num_experts):
            self.mats.append(torch.nn.Parameter(torch.randn(input_dim, experts_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(experts_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(experts_hidden, experts_out)))
            self.mats.append(torch.nn.Parameter(torch.randn(experts_out)))
        # gates
        for i in range(tasks):
            self.mats.append(torch.nn.Parameter(torch.randn(input_dim, num_experts)))
        # towers
        for i in range(tasks):
            self.mats.append(torch.nn.Parameter(torch.randn(experts_out, towers_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(towers_hidden)))
            self.mats.append(torch.nn.Parameter(torch.randn(towers_hidden, towers_out)))
            self.mats.append(torch.nn.Parameter(torch.randn(towers_out)))

    def experts_o(self, batch_size, index, feats, values, experts):
        for i in range(self.num_experts):
            params = experts[i * 4 : i * 4 + 4]
            # 构造 sparse matrix
            # torch.sparse.mm

    def forward_(self, batch_size, index, feats, values, bias, weights, embeddings, mats: List[torch.Tensor], fields=torch.Tensor([])):
        experts, gates, towers = self.parse_mats(mats)

        # experts_o = [e(x) for e in self.experts]
        # experts_o_tensor = torch.stack(experts_o)

        # gates_o = [self.softmax(x @ g) for g in self.w_gates]

        # tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        # tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        # # final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        # return self.towers[0](tower_input[0])

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

    def forward(self, batch_size, index, feats, values, fields=torch.Tensor([])):
        # batch: coo
        # index = torch.from_numpy(batch.row).to(torch.long)
        # feats = torch.from_numpy(batch.col).to(torch.long)
        # values = torch.from_numpy(batch.data)
        return self.forward_(batch_size, index, feats, values, self.bias, self.weights, self.embeddings, self.mats, fields)

    def parse_mats(self, mats: List[torch.Tensor]):
        a = 4 * self.num_experts
        b = self.tasks
        # c = 4 * self.tasks
        return mats[:a], mats[a : a + b], mats[a + b :]


if __name__ == "__main__":
    model = MMoE(input_dim=148, n_fields=-1, embedding_dim=1, experts_hidden=32, experts_out=16, towers_hidden=8, towers_out=1, num_experts=6, tasks=1)
    script_module = torch.jit.script(model)
    script_module.save("MMoE.pt")
    print(model)
