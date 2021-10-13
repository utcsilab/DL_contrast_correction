import torch
from torch import nn
from torch.nn import functional as F
# From fMRI
class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()

# following from this link https://learnopencv.com/paired-image-to-image-translation-pix2pix/
adversarial_loss = nn.BCELoss() #wgan loss is also a good option
l1_loss = nn.L1Loss()


def generator_loss(generated_image, target_img, G, real_target, Lambda = 100):
    # G: Output predictions from the discriminator, when fed with generator-produced images.
    gen_loss = adversarial_loss(G, real_target)
    loss_val = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (Lambda* loss_val)
    # print(gen_loss)
    return gen_total_loss

def generator_loss_separately(generated_image, target_img, G, real_target, Lambda = 100):
    # G: Output predictions from the discriminator, when fed with generator-produced images.
    # similar to the above function, just it gives 2 outputs
    adv_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    return l1_l, adv_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

class NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        error_norm = torch.square(torch.norm(X - Y))
        self_norm  = torch.square(torch.norm(X))

        return torch.sqrt(error_norm / self_norm)
