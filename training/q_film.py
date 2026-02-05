
'''CLASS IS INCOMPLETE, STILL A WORK IN PROGRESS. DO NOT TRY TO USE YET'''
class FiLMDuelQNet(nn.Module):
    """
    FiLM: vars -> (gamma,beta) used to modulate CNN activations.
    Uses vars only through modulation (no late concat).
    """
    def __init__(self, available_actions_count: int, num_vars: int):
        super().__init__()
        # channel sizes per conv block
        c1, c2, c3, c4 = 8, 8, 8, 16
        self.c = (c1, c2, c3, c4)

        # conv blocks (keep BN, but apply FiLM before ReLU)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)

        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(c4)

        # vars -> gammas/betas for each block
        total = 2 * sum(self.c)  # gamma+beta for each channel across all blocks
        self.film_mlp = nn.Sequential(
            nn.LayerNorm(num_vars),
            nn.Linear(num_vars, 128),
            nn.ReLU(),
            nn.Linear(128, total),
        )
        # initialize last layer to 0 so FiLM starts ~identity
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # dueling heads on flattened conv output
        self.state_fc = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_fc = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

    def _split_film(self, film: torch.Tensor):
        # film: (B, 2*sumC)
        B = film.size(0)
        parts = []
        idx = 0
        for ch in self.c:
            gamma = film[:, idx : idx + ch]
            beta = film[:, idx + ch : idx + 2 * ch]
            parts.append((gamma, beta))
            idx += 2 * ch
        return parts  # list of (gamma,beta) per block

    def forward(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        film = self.film_mlp(vars_)
        (g1,b1), (g2,b2), (g3,b3), (g4,b4) = self._split_film(film)

        x = self.conv1(img)
        x = self.bn1(x)
        x = _apply_film(x, g1, b1)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = _apply_film(x, g2, b2)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = _apply_film(x, g3, b3)
        x = torch.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = _apply_film(x, g4, b4)
        x = torch.relu(x)

        x = x.view(x.size(0), -1)  # (B, 192)

        state_value = self.state_fc(x)
        advantage = self.advantage_fc(x)
        q = state_value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
