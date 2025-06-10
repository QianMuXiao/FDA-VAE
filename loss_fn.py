import torch
from generative.losses import PatchAdversarialLoss

intensity_loss = torch.nn.L1Loss()
adv_loss = PatchAdversarialLoss(criterion="bce")

adv_weight = 0.1
perceptual_weight = 0.1
# kl_weight: important hyper-parameter.
#     If too large, decoder cannot recon good results from latent space.
#     If too small, latent space will not be regularized enough for the diffusion model
kl_weight = 1e-7

bce_loss = torch.nn.BCEWithLogitsLoss()

def compute_kl_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
    return torch.sum(kl_loss) / kl_loss.shape[0]

def generator_loss(gen_images, real_images, z_mu, z_sigma, disc_net, loss_perceptual):
    recons_loss = intensity_loss(gen_images, real_images)
    kl_loss = compute_kl_loss(z_mu, z_sigma)
    p_loss = loss_perceptual(gen_images.float(), real_images.float())
    loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

    logits_fake = disc_net(gen_images)[-1]
    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    loss_g = loss_g + adv_weight * generator_loss

    return loss_g


def mean_symmetry_loss(mu1, mu2, lambda_distance=1.0, alpha_opposite=1.0):
    """
    Compute mean symmetry loss with additional constraints.
    Args:
        mu1: Mean of distribution 1.
        mu2: Mean of distribution 2.
        lambda_distance: Weight for distance equality constraint.
        alpha_opposite: Weight for opposite direction constraint.
    Returns:
        Symmetry loss for means.
    """
    # Original symmetry loss
    mean_sum_loss = torch.norm(mu1 + mu2, p=2)  # ||mu1 + mu2||_2

    # Distance equality constraint: ||mu1||_2 = ||mu2||_2
    mean_distance_loss = torch.abs(torch.norm(mu1, p=2) - torch.norm(mu2, p=2))

    # Opposite direction constraint: mu1 * mu2 <= 0
    opposite_direction_loss = torch.min(torch.zeros_like(mu1), mu1 * mu2).sum()

    # Combine losses
    total_loss = mean_sum_loss + lambda_distance * mean_distance_loss - alpha_opposite * opposite_direction_loss
    return total_loss


def discriminator_loss(gen_images, real_images, disc_net):
    gen_images = gen_images.to(torch.float32)
    # real_images = gen_images.to(torch.float32)
    
    
    logits_fake = disc_net(gen_images.contiguous().detach())[-1]
    
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
    
    logits_real = disc_net(real_images.contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
    d_loss = (loss_d_fake + loss_d_real) * 0.5
    
    return d_loss, loss_d_fake, loss_d_real