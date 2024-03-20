import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *


###########################################################################################
## Model
###########################################################################################
class Generator(nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        z = latent_size
        d = output_size
        self.main = nn.Sequential(
            nn.Linear(z, 2 * latent_size), nn.ReLU(), nn.Linear(2 * latent_size, d)
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, wasserstein=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, int(input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(input_size / 2), 1),
        )

        if not wasserstein:
            self.main.add_module(str(3), nn.Sigmoid())

    def forward(self, x):
        return self.main(x)

    ###########################################################################################


## Train and Generate
###########################################################################################
class DP_WGAN:
    def __init__(
        self,
        input_dim,
        z_dim,
        y_dim,
        dset_size,
        data_loader,
        generator,
        discriminator,
        target_epsilon,
        target_delta,
        hyperparams,
        conditional=True,
        onehot=True,
    ):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.dset_size = dset_size
        self.data_loader = data_loader
        self.generator = generator
        self.discriminator = discriminator
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.conditional = conditional
        self.onehot = onehot
        self.hyperparams = hyperparams
        self.optimizer_g = optim.RMSprop(self.generator.parameters(), lr=hyperparams.lr)
        self.optimizer_d = optim.RMSprop(
            self.discriminator.parameters(), lr=hyperparams.lr
        )

    def train(self, curr_stats, private=False, conditional=True):
        one = torch.cuda.DoubleTensor([1])
        mone = one * -1

        batch_size = self.hyperparams.batch_size
        micro_batch_size = self.hyperparams.micro_batch_size
        clamp_upper = self.hyperparams.clamp_upper
        clamp_lower = self.hyperparams.clamp_lower
        clip_coeff = self.hyperparams.clip_coeff
        sigma = self.hyperparams.sigma
        class_ratios = torch.from_numpy(self.hyperparams.class_ratios)

        epsilon = curr_stats.epsilon
        gen_iters = curr_stats.gen_iters
        steps = curr_stats.steps
        epoch = curr_stats.epoch

        data_iter = iter(self.data_loader)
        i = 0
        while i < len(self.data_loader):
            # Update Critic
            for p in self.discriminator.parameters():
                p.requires_grad = True

            if gen_iters < 25 or gen_iters % 500 == 0:
                disc_iters = 100

            else:
                disc_iters = 5

            j = 0
            while j < disc_iters and i < len(self.data_loader):
                j += 1

                # clamp parameters to a cube
                for p in self.discriminator.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

                # data = data_iter.next()
                data = next(data_iter)
                i += 1

                # train with real
                self.optimizer_d.zero_grad()
                inputs, categories = data
                inputs, categories = inputs.cuda(), categories.cuda()

                if self.onehot:
                    categories = one_hot_embedding(categories, num_classes=self.y_dim)
                    err_d_real = self.discriminator(
                        torch.cat([inputs, categories.cuda().double()], dim=1)
                    )

                else:
                    err_d_real = self.discriminator(
                        torch.cat(
                            [inputs, categories.cuda().unsqueeze(1).double()], dim=1
                        )
                    )

                if private:
                    # For privacy, clip the avg gradient of each micro-batch
                    clipped_grads = {
                        name: torch.zeros_like(param)
                        for name, param in self.discriminator.named_parameters()
                    }

                    for k in range(int(err_d_real.size(0) / micro_batch_size)):
                        err_micro = (
                            err_d_real[
                                k * micro_batch_size : (k + 1) * micro_batch_size
                            ]
                            .mean(0)
                            .view(1)
                        )
                        err_micro.backward(one, retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), clip_coeff
                        )
                        for name, param in self.discriminator.named_parameters():
                            clipped_grads[name] += param.grad
                        self.discriminator.zero_grad()

                    for name, param in self.discriminator.named_parameters():
                        # add noise here
                        param.grad = (
                            clipped_grads[name]
                            + torch.DoubleTensor(clipped_grads[name].size())
                            .normal_(0, sigma * clip_coeff)
                            .cuda()
                        ) / (err_d_real.size(0) / micro_batch_size)

                    steps += 1

                else:
                    err_d_real.mean(0).view(1).backward(one)

                # train with fake
                noise = torch.randn(batch_size, self.z_dim).cuda()
                if conditional:
                    category = (
                        torch.multinomial(class_ratios, batch_size, replacement=True)
                        .cuda()
                        .double()
                    )
                    if self.onehot:
                        category = one_hot_embedding(category, num_classes=self.y_dim)
                    else:
                        category = category.unsqueeze(1)
                    category = category.cuda()
                    fake = self.generator(torch.cat([noise.double(), category], dim=1))
                    err_d_fake = (
                        self.discriminator(torch.cat([fake.detach(), category], dim=1))
                        .mean(0)
                        .view(1)
                    )

                else:
                    fake = self.generator(noise.double())
                    err_d_fake = self.discriminator(fake.detach()).mean(0).view(1)
                err_d_fake.backward(mone)
                self.optimizer_d.step()

            # Update Generator
            for p in self.discriminator.parameters():
                p.requires_grad = False

            self.optimizer_g.zero_grad()
            noise = torch.randn(batch_size, self.z_dim).cuda()
            if conditional:
                category = (
                    torch.multinomial(class_ratios, batch_size, replacement=True)
                    .cuda()
                    .double()
                )
                if self.onehot:
                    category = one_hot_embedding(category, num_classes=self.y_dim)
                else:
                    category = category.unsqueeze(1)
                category = category.cuda()
                fake = self.generator(torch.cat([noise.double(), category], dim=1))
                err_g = (
                    self.discriminator(torch.cat([fake, category.double()], dim=1))
                    .mean(0)
                    .view(1)
                )
            else:
                fake = self.generator(noise.double())
                err_g = self.discriminator(fake).mean(0).view(1)
            err_g.backward(one)
            self.optimizer_g.step()
            gen_iters += 1

        epoch += 1
        if private:
            sampling_prob = batch_size / float(self.dset_size)
            epsilon = compute_epsilon(
                sigma, sampling_prob, steps, delta=self.target_delta
            )
        else:
            if epoch > self.hyperparams.num_epochs:
                epsilon = np.inf
        print(
            "Epoch :",
            epoch,
            "Loss D real : ",
            err_d_real.mean(0).view(1).item(),
            "Loss D fake : ",
            err_d_fake.item(),
            "Loss G : ",
            err_g.item(),
            "Epsilon spent : ",
            epsilon,
        )
        return epsilon, epoch, steps, gen_iters

    def generate(self, num_rows, class_ratios, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim).cuda()
            if self.conditional:
                cat = (
                    torch.multinomial(class_ratios, batch_size, replacement=True)
                    .cuda()
                    .double()
                )
                if self.onehot:
                    cat_ = one_hot_embedding(cat, num_classes=self.y_dim)
                else:
                    cat_ = cat.unsqueeze(1)
                synthetic = self.generator(torch.cat([noise.double(), cat_], dim=1))
                synthetic = torch.cat([synthetic, cat.unsqueeze(1)], dim=1)

            else:
                synthetic = self.generator(noise.double())

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps * batch_size < num_rows:
            noise = torch.randn(num_rows - steps * batch_size, self.z_dim).cuda()

            if self.conditional:
                cat = (
                    torch.multinomial(
                        class_ratios, num_rows - steps * batch_size, replacement=True
                    )
                    .cuda()
                    .double()
                )
                if self.onehot:
                    cat_ = one_hot_embedding(cat, num_classes=self.y_dim)
                else:
                    cat_ = cat.unsqueeze(1)
                cat_ = cat_.cuda()
                synthetic = self.generator(torch.cat([noise.double(), cat_], dim=1))
                synthetic = torch.cat([synthetic, cat.unsqueeze(1)], dim=1)
            else:
                synthetic = self.generator(noise.double())
            synthetic_data.append(synthetic.cpu().data.numpy())
        return np.concatenate(synthetic_data)
