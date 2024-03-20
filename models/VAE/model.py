import torch
import torch.nn as nn
import numpy as np

from utils import one_hot_embedding


class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, beta=1, transform="none"):
        super(CVAE, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.beta = beta
        self.transform = transform

        self.fc_feat_x = nn.Sequential(
            nn.Linear(x_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_feat_y = nn.Sequential(nn.Linear(y_dim, 256), nn.ReLU())
        self.fc_feat_all = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        self.dec_z = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU())
        self.dec_y = nn.Sequential(nn.Linear(y_dim, 256), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Linear(1000, x_dim),
        )

        self.rec_crit = nn.MSELoss()
        self.encoder = [
            self.fc_feat_x,
            self.fc_feat_y,
            self.fc_feat_all,
            self.fc_logvar,
            self.fc_mu,
        ]
        self.decoder = [self.dec_z, self.dec_y, self.dec]
        self.encoder_params = list()
        for layer in self.encoder:
            self.encoder_params = self.encoder_params + list(layer.parameters())
        self.decoder_params = list()
        for layer in self.decoder:
            self.decoder_params = self.decoder_params + list(layer.parameters())

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        out = self.decode(self.reparameterize(mu, logvar), y)

        return mu, logvar, out

    def compute_loss(self, x, y, verbose=True):
        mu, logvar, rec = self.forward(x, y)

        rec_loss = self.rec_crit(rec, x)

        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = rec_loss + self.beta * kl_loss

        if verbose:
            return {
                "loss": loss,
                "rec_loss": rec_loss,
                "kl_loss": kl_loss,
                "mu": mu,
                "logvar": logvar,
                "rec": rec,
            }
        else:
            return {"loss": loss, "rec_loss": rec_loss, "kl_loss": kl_loss}

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x, y):
        feat_x = self.fc_feat_x(x)
        feat_y = self.fc_feat_y(y)
        feat = torch.cat([feat_x, feat_y], dim=1)
        feat = self.fc_feat_all(feat)

        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)

        return mu, logvar

    def decode(self, z, y):
        dec_y = self.dec_y(y)
        dec_z = self.dec_z(z)
        out = self.dec(torch.cat([dec_z, dec_y], dim=1))

        if self.transform == "exp":
            out = out.exp()
        elif self.transform == "tahn":
            out = torch.tahn(out)
        elif self.transform == "sigmoid":
            out = torch.sigmoid(out)
        elif self.transform == "relu":
            out = torch.nn.ReLU()(out)
        return out

    def sample(self, n, data_loader, uniform_y=False, device="cpu"):
        """Returns (fake_data, fake_label) samples."""

        self.eval()

        fake_data = []
        fake_label = []
        total = 0

        while total <= n:
            for data_x, data_y in data_loader:

                if total > n:
                    break
                else:
                    z = torch.randn([data_x.shape[0], self.z_dim]).to(device)
                    if uniform_y:
                        data_y = torch.randint(0, self.y_dim, [data_x.shape[0]])
                    y = one_hot_embedding(data_y, num_classes=self.y_dim, device=device)
                fake_data.append(self.decode(z, y).detach().cpu().numpy())
                fake_label.append(data_y.cpu().numpy())

                total += len(data_x)

        fake_data = np.concatenate(fake_data)[:n]
        fake_label = np.concatenate(fake_label)[:n]

        return fake_data, fake_label
