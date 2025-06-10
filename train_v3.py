import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler as DSampler

from tqdm import tqdm
from tensorboardX import SummaryWriter

from mri_dataset_v2 import MultiModalDataset
from Yshape_AutoencoderKL import AutoencoderKL
from generative.networks.nets import PatchDiscriminator
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from loss_fn import compute_kl_loss, discriminator_loss
from pytorch_msssim import ssim
from lpips import LPIPS


import time
from collections import deque

def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

intensity_loss = torch.nn.L1Loss()

def tahn2sigmoid(input):
    return (input + 1) / 2

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)  
    device = torch.device(f'cuda:{local_rank}')
    return device



def get_available_gpus():
    return torch.cuda.device_count()

#这个训练函数中，训练在多GPU上训练，验证仅在住进程中验证

def train_VAEYnet(
    train_loader,
    train_sampler,
    val_loader,
    val_sampler,
    generator, 
    disc_ct,
    disc_mr,
    opt_gen,
    opt_ct,
    opt_mr,
    scheduler_gen,
    scheduler_disc_ct,
    scheduler_disc_mr,
    d_train_freq = 1,
    device = None,
    Max_epoch = 1501,
    writer = None,
    model_save_path = None,
    model_save_interval = 10,
    perceptual_loss = None,
    adv_loss = None,
    weight_kl = 1e-7,
    weight_perceptual = 0.01,
    weight_adv = 0.01,
    weight_recons = 0.01,
    weight_cycle = 1,
    weight_diff = 0.01,
    lpips = None
):
    
    try:
        # epoch_avg_gan_loss_ct = []
        # epoch_avg_gan_loss_mr = []
        # epoch_avg_disc_loss_ct = []
        # epoch_avg_disc_loss_mr = []
        
        recon_loss_history = deque(maxlen=20)
        
        best_mr2ct_psnr = 0

        
        for epoch_nums in range(Max_epoch):
            #training loop

            train_sampler.set_epoch(epoch_nums)

            generator.train()
            if dist.get_rank() == 0:
                pbar = tqdm(train_loader, total=len(train_loader), desc='Epoch %d'%epoch_nums, dynamic_ncols=True)
            else:
                pbar = train_loader

            

            for batch_idx, (real_delay, real_pre, _, _) in enumerate(pbar):
                
                real_ct_img = real_delay.to(device)
                real_mr_img = real_pre.to(device)
                
                ct2ct, ct2mr, z_mu_ct, z_sigma_ct = generator(real_ct_img, label='ct')
                mr2ct, mr2mr, z_mu_mr, z_sigma_mr = generator(real_mr_img, label='mr')
                
                
                recons_loss_ct2ct = intensity_loss(ct2ct, real_ct_img)
                recons_loss_mr2mr = intensity_loss(mr2mr, real_mr_img)
                recon_loss = (recons_loss_ct2ct + recons_loss_mr2mr) * 0.5
                
                if epoch_nums >= 0 and len(recon_loss_history) > 5:

                    history_mean = sum(recon_loss_history) / len(recon_loss_history)
                    threshold = history_mean * 2

                    is_abnormal = torch.tensor(
                        [float(recon_loss.item() > threshold)],
                        device=device
                    )
                    dist.all_reduce(is_abnormal, op=dist.ReduceOp.MAX)

                    if is_abnormal.item() > 0.5:
                        recon_loss_history.append(recon_loss.item())
                        if dist.get_rank() == 0:
                            pbar.set_description(f"Epoch {epoch_nums} [WARNING: skip update for batch {batch_idx}]")

                        del recon_loss, ct2ct, ct2mr, mr2ct, mr2mr, z_mu_ct, z_sigma_ct, z_mu_mr, z_sigma_mr, real_ct_img, real_mr_img, recons_loss_ct2ct, recons_loss_mr2mr
                        torch.cuda.empty_cache()
                        dist.barrier()
                        time.sleep(0.5)
                        
                        
                        opt_gen.zero_grad()
                        continue
                        # pass
                    else:
                        pass
                
                    
                    
                    
                    
                    
                # train discriminator
                d_total_loss_ct = torch.zeros(1).to(device)
                d_total_loss_ct_real = torch.zeros(1).to(device)
                d_total_loss_ct_fake = torch.zeros(1).to(device)
                d_total_loss_mr = torch.zeros(1).to(device)
                d_total_loss_mr_real = torch.zeros(1).to(device)
                d_total_loss_mr_fake = torch.zeros(1).to(device)
                    
                for _ in range(d_train_freq):
                    # train discriminator for ct

                    d_mr2ct_loss, d_mr2ct_fake, d_mr2ct_real = discriminator_loss(
                        mr2ct, real_ct_img, disc_net=disc_ct
                    )
                    
                    d_ct_loss = d_mr2ct_loss 
                    d_ct_fake = d_mr2ct_fake 
                    d_ct_real = d_mr2ct_real 
                    
                    
                    opt_ct.zero_grad()
                    d_ct_loss.backward()
                    opt_ct.step()

                
                    d_total_loss_ct += d_ct_loss.item()
                    d_total_loss_ct_fake += d_ct_fake.item()
                    d_total_loss_ct_real += d_ct_real.item()
                    
                    # train discriminator for mr


                    d_ct2mr_loss, d_ct2mr_fake, d_ct2mr_real = discriminator_loss(
                        ct2mr, real_mr_img, disc_net=disc_mr
                    )

                    
                    d_mr_loss = d_ct2mr_loss 
                    d_mr_fake = d_ct2mr_fake 
                    d_mr_real = d_ct2mr_real
                    
                    
                    
                    opt_mr.zero_grad()
                    d_mr_loss.backward()
                    opt_mr.step()

                    
                    
                    d_total_loss_mr += d_mr_loss.item()
                    d_total_loss_mr_fake += d_mr_fake.item()
                    d_total_loss_mr_real += d_mr_real.item()
                

                kl_ct = compute_kl_loss(z_mu_ct, z_sigma_ct)
                kl_mr = compute_kl_loss(z_mu_mr, z_sigma_mr)
                kl_loss = (kl_ct + kl_mr) * 0.5
                
                trans_loss_mr2ct = intensity_loss(mr2ct, real_ct_img)
                trans_loss_ct2mr = intensity_loss(ct2mr, real_mr_img)
                trans_loss = (trans_loss_mr2ct + trans_loss_ct2mr) * 0.5
                
                logits_fake_mr2ct = disc_ct(mr2ct)[-1]
                logits_fake_ct2mr = disc_mr(ct2mr)[-1]

                gan_loss_mr2ct = adv_loss(logits_fake_mr2ct, target_is_real = True, for_discriminator=False)
                gan_loss_ct2mr = adv_loss(logits_fake_ct2mr, target_is_real = True, for_discriminator=False)
                
                gan_loss = (gan_loss_mr2ct + gan_loss_ct2mr) * 0.5

                
                
                

                perce_ct2mr = perceptual_loss(ct2mr.float(), real_mr_img.float())
                perce_mr2ct = perceptual_loss(mr2ct.float(), real_ct_img.float())

                
                perce_loss = (perce_ct2mr + perce_mr2ct) * 0.5

                
                
                
                kl_diff_mu = intensity_loss(z_mu_ct, -z_mu_mr)
                kl_diff_sigma = intensity_loss(z_sigma_ct.pow(2), z_sigma_mr.pow(2))
                kl_diff = kl_diff_mu/2 + kl_diff_sigma


                g_loss = (
                    weight_adv * gan_loss +
                    weight_recons * recon_loss +
                    weight_cycle * trans_loss +
                    weight_kl * kl_loss +
                    weight_diff * kl_diff +
                    weight_perceptual * perce_loss
                )
                    
                opt_gen.zero_grad()
                g_loss.backward()
                opt_gen.step()

                
                
                
                recon_loss_history.append(recon_loss.item())
                
                losses_to_average = {
                    'g_loss': g_loss.detach(),
                    'd_loss_ct': d_total_loss_ct / d_train_freq,
                    'd_loss_mr': d_total_loss_mr / d_train_freq,
                    'gan_loss': gan_loss.detach(),
                    'recon_loss': recon_loss.detach(),
                    'cycle_loss': trans_loss.detach(),
                    'recons_loss_ct2ct': recons_loss_ct2ct.detach(),
                    'recons_loss_mr2mr': recons_loss_mr2mr.detach(),
                    'trans_loss_mr2ct': trans_loss_mr2ct.detach(),
                    'trans_loss_ct2mr': trans_loss_ct2mr.detach(),
                    'gan_loss_ct2mr': gan_loss_ct2mr.detach(),
                    'gan_loss_mr2ct': gan_loss_mr2ct.detach(),
                    'perce_loss': perce_loss.detach(),
                    'perce_ct2mr': perce_ct2mr.detach(),
                    'perce_mr2ct': perce_mr2ct.detach(),
                    'kl_loss': kl_loss.detach(),
                    'kl_ct': kl_ct.detach(),
                    'kl_mr': kl_mr.detach(),
                    'kl_diff_mu': kl_diff_mu.detach(),
                    'kl_diff_sigma': kl_diff_sigma.detach(),
                    'd_loss_ct_real': d_total_loss_ct_real / d_train_freq,
                    'd_loss_ct_fake': d_total_loss_ct_fake / d_train_freq,
                    'd_loss_mr_real': d_total_loss_mr_real / d_train_freq,
                    'd_loss_mr_fake': d_total_loss_mr_fake / d_train_freq,
                }

                for key in losses_to_average:
                    losses_to_average[key] = losses_to_average[key].clone()
                    torch.distributed.all_reduce(losses_to_average[key], op=torch.distributed.ReduceOp.SUM)
                    losses_to_average[key] /= dist.get_world_size()

                if dist.get_rank() == 0:
                    pbar.set_postfix({
                        'g_loss': losses_to_average['g_loss'].item(),
                        'd_loss_ct': losses_to_average['d_loss_ct'].item(),
                        'd_loss_mr': losses_to_average['d_loss_mr'].item(),
                        'gan_loss': losses_to_average['gan_loss'].item(),
                        'recon_loss': losses_to_average['recon_loss'].item(),
                        'cycle_loss': losses_to_average['cycle_loss'].item(),
                    })

                    if writer:
                        global_step = epoch_nums * len(train_loader) + batch_idx

                        
                        writer.add_scalar('Train_Loss/generator', losses_to_average['g_loss'].item(), global_step)
                        current_lr = optimizer_gen.param_groups[0]['lr']
                        writer.add_scalar('Learning_rate/generator', current_lr, epoch_nums)
                        
                        writer.add_scalars(
                            'Train_Loss/l1_loss_parts', {
                                'rebuild': losses_to_average['recon_loss'].item(),
                                'transform': losses_to_average['cycle_loss'].item()
                            }, global_step
                        )

                        writer.add_scalars(
                            'Train_Loss/recon_loss_parts', {
                                'CT2CT': losses_to_average['recons_loss_ct2ct'].item(),
                                'MR2MR': losses_to_average['recons_loss_mr2mr'].item(),
                                'MR2CT': losses_to_average['trans_loss_mr2ct'].item(),
                                'CT2MR': losses_to_average['trans_loss_ct2mr'].item()
                            }, global_step
                        )

                        writer.add_scalar('Train_Loss/gan_loss', losses_to_average['gan_loss'].item(), global_step)
                        writer.add_scalars(
                            'Train_Loss/gan_loss_parts', {
                                'CT2MRI': losses_to_average['gan_loss_ct2mr'].item(),
                                'MRI2CT': losses_to_average['gan_loss_mr2ct'].item(),
                            }, global_step
                        )

                        writer.add_scalar('Train_Loss/perce_loss', losses_to_average['perce_loss'].item(), global_step)
                        writer.add_scalars(
                            'Train_Loss/perce_loss_parts', {
                                'CT2MRI': losses_to_average['perce_ct2mr'].item(),
                                'MRI2CT': losses_to_average['perce_mr2ct'].item(),
                            }, global_step
                        )

                        writer.add_scalar('Train_Loss/kl_loss', losses_to_average['kl_loss'].item(), global_step)
                        writer.add_scalars(
                            'Train_Loss/kl_loss_parts', {
                                'CT': losses_to_average['kl_ct'].item(),
                                'MRI': losses_to_average['kl_mr'].item()
                            }, global_step
                        )

                        writer.add_scalars(
                            'Train_Loss/kl_diff_parts', {
                                'mu': losses_to_average['kl_diff_mu'].item(),
                                'sigma': losses_to_average['kl_diff_sigma'].item()
                            }, global_step
                        )

                        writer.add_scalar('Train_Loss/discriminator_ct', losses_to_average['d_loss_ct'].item(), global_step)
                        writer.add_scalar('Train_Loss/discriminator_mr', losses_to_average['d_loss_mr'].item(), global_step)
                        writer.add_scalars(
                            'Train_Loss/discriminator_ct_parts', {
                                'Real': losses_to_average['d_loss_ct_real'].item(),
                                'Fake': losses_to_average['d_loss_ct_fake'].item()
                            }, global_step
                        )
                        writer.add_scalars(
                            'Train_Loss/discriminator_mr_parts', {
                                'Real': losses_to_average['d_loss_mr_real'].item(),
                                'Fake': losses_to_average['d_loss_mr_fake'].item()
                            }, global_step
                        )

            scheduler_gen.step()
            scheduler_disc_ct.step()
            scheduler_disc_mr.step()
            
            torch.distributed.barrier()
            generator.eval()
            
            # Initialize metrics
            val_ct2ct_loss = 0 
            val_ct2mr_loss = 0
            val_mr2ct_loss = 0
            val_mr2mr_loss = 0

            val_ct2ct_ssim = 0
            val_ct2mr_ssim = 0
            val_mr2ct_ssim = 0
            val_mr2mr_ssim = 0

            val_percetual_ct2mr = 0
            val_percetual_mr2ct = 0

            val_psnr_ct = 0
            val_psnr_ct2ct = 0
            val_psnr_mr = 0
            val_psnr_mr2mr = 0

            val_ct2mr_lpips = 0
            val_mr2ct_lpips = 0
            
            total_samples = 0



            val_sampler.set_epoch(epoch_nums)

            if dist.get_rank() == 0:
                pbar = tqdm(val_loader, total=len(val_loader), desc='Epoch %d'%epoch_nums, dynamic_ncols=True)
            else:
                pbar = val_loader


            with torch.no_grad():

                for batch_idx, (real_delay, real_pre, _, _) in enumerate(pbar):
                    real_ct_img = real_delay.to(device)
                    real_mr_img = real_pre.to(device)
                    
                    batch_size = real_ct_img.size(0)
                    total_samples += batch_size

                    
                    ct2ct, ct2mr, _, _ = generator.module(real_ct_img, label = 'ct')
                    mr2ct, mr2mr, _, _ = generator.module(real_mr_img, label = 'mr')
                    
                    ct2ct_recons_loss = intensity_loss(ct2ct, real_ct_img)
                    ct2mr_recons_loss = intensity_loss(ct2mr, real_mr_img)
                    mr2ct_recons_loss = intensity_loss(mr2ct, real_ct_img)
                    mr2mr_recons_loss = intensity_loss(mr2mr, real_mr_img)

                    ct2mr_perce = perceptual_loss(ct2mr.float(), real_mr_img.float())
                    mr2ct_perce = perceptual_loss(mr2ct.float(), real_ct_img.float())
                    
                    ct2mr_lpips = lpips(ct2mr.float(), real_mr_img.float())
                    mr2ct_lpips = lpips(mr2ct.float(), real_ct_img.float())
                    
                    
                    
                    real_ct_img = tahn2sigmoid(real_ct_img)
                    real_mr_img = tahn2sigmoid(real_mr_img)
                    ct2ct = tahn2sigmoid(ct2ct)
                    ct2mr = tahn2sigmoid(ct2mr)
                    mr2ct = tahn2sigmoid(mr2ct)
                    mr2mr = tahn2sigmoid(mr2mr)
                    
                    
                    
                    ct2ct_ssim = ssim(ct2ct, real_ct_img, data_range=1.0, size_average=True, win_size=11)
                    ct2mr_ssim = ssim(ct2mr, real_mr_img, data_range=1.0, size_average=True, win_size=11)
                    
                    mr2ct_ssim = ssim(mr2ct, real_ct_img, data_range=1.0, size_average=True, win_size=11)
                    mr2mr_ssim = ssim(mr2mr, real_mr_img, data_range=1.0, size_average=True, win_size=11)
                    
                    ct2ct_psnr = calculate_psnr(ct2ct, real_ct_img)
                    ct2mr_psnr = calculate_psnr(ct2mr, real_mr_img)
                    
                    mr2mr_psnr = calculate_psnr(mr2mr, real_mr_img)
                    mr2ct_psnr = calculate_psnr(mr2ct, real_ct_img)
                    

                    val_ct2ct_loss += ct2ct_recons_loss.item() * batch_size
                    val_ct2mr_loss += ct2mr_recons_loss.item() * batch_size
                    val_mr2ct_loss += mr2ct_recons_loss.item() * batch_size
                    val_mr2mr_loss += mr2mr_recons_loss.item() * batch_size
                    
                    val_ct2mr_ssim += ct2mr_ssim.item() * batch_size
                    val_ct2ct_ssim += ct2ct_ssim.item() * batch_size
                    val_mr2ct_ssim += mr2ct_ssim.item() * batch_size
                    val_mr2mr_ssim += mr2mr_ssim.item() * batch_size
                    
                    val_psnr_ct += mr2ct_psnr.item() * batch_size
                    val_psnr_ct2ct += ct2ct_psnr.item() * batch_size
                    val_psnr_mr += ct2mr_psnr.item() * batch_size
                    val_psnr_mr2mr += mr2mr_psnr.item() * batch_size
                    
                    val_percetual_ct2mr += ct2mr_perce.item() * batch_size
                    val_percetual_mr2ct += mr2ct_perce.item() * batch_size
                    
                    val_ct2mr_lpips += ct2mr_lpips.mean().item() * batch_size
                    val_mr2ct_lpips += mr2ct_lpips.mean().item() * batch_size

            metrics = torch.tensor([
                val_ct2ct_loss,
                val_ct2mr_loss,
                val_mr2ct_loss,
                val_mr2mr_loss,
                val_ct2mr_ssim,
                val_ct2ct_ssim,
                val_mr2ct_ssim,
                val_mr2mr_ssim,
                val_psnr_ct,
                val_psnr_ct2ct,
                val_psnr_mr,
                val_psnr_mr2mr,
                val_percetual_ct2mr,
                val_percetual_mr2ct,
                val_ct2mr_lpips,
                val_mr2ct_lpips
            ]).to(device)


            total_samples_tensor = torch.tensor(total_samples).to(device)
            torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
            total_samples = total_samples_tensor.item()

            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)


            (
                val_ct2ct_loss,
                val_ct2mr_loss,
                val_mr2ct_loss,
                val_mr2mr_loss,
                val_ct2mr_ssim,
                val_ct2ct_ssim,
                val_mr2ct_ssim,
                val_mr2mr_ssim,
                val_psnr_ct,
                val_psnr_ct2ct,
                val_psnr_mr,
                val_psnr_mr2mr,
                val_percetual_ct2mr,
                val_percetual_mr2ct,
                val_ct2mr_lpips,
                val_mr2ct_lpips
            ) = metrics.tolist()

            avg_val_ct2ct_loss = val_ct2ct_loss / total_samples
            avg_val_ct2mr_loss = val_ct2mr_loss / total_samples
            avg_val_mr2ct_loss = val_mr2ct_loss / total_samples
            avg_val_mr2mr_loss = val_mr2mr_loss / total_samples

            avg_val_ct2mr_ssim = val_ct2mr_ssim / total_samples
            avg_val_ct2ct_ssim = val_ct2ct_ssim / total_samples
            avg_val_mr2ct_ssim = val_mr2ct_ssim / total_samples
            avg_val_mr2mr_ssim = val_mr2mr_ssim / total_samples

            avg_val_psnr_ct = val_psnr_ct / total_samples
            avg_val_psnr_ct2ct = val_psnr_ct2ct / total_samples
            avg_val_psnr_mr = val_psnr_mr / total_samples
            avg_val_psnr_mr2mr = val_psnr_mr2mr / total_samples

            avg_val_percetual_ct2mr = val_percetual_ct2mr / total_samples
            avg_val_percetual_mr2ct = val_percetual_mr2ct / total_samples
            
            avg_val_lpips_ct2mr = val_ct2mr_lpips / total_samples
            avg_val_lpips_mr2ct = val_mr2ct_lpips / total_samples

            if writer and dist.get_rank() == 0:
                writer.add_scalars(
                    'Val_loss/l1_loss_parts',{
                        'CT2CT':avg_val_ct2ct_loss,
                        'CT2MR':avg_val_ct2mr_loss,
                        'MR2CT':avg_val_mr2ct_loss,
                        'MR2MR':avg_val_mr2mr_loss
                    }, epoch_nums
                )

                writer.add_scalars(
                    'Val_loss/SSIM_res_parts',{
                        'CT2CT':avg_val_ct2ct_ssim,
                        'CT2MR':avg_val_ct2mr_ssim,
                        'MR2CT':avg_val_mr2ct_ssim,
                        'MR2MR':avg_val_mr2mr_ssim
                    }, epoch_nums
                )
                
                writer.add_scalars(
                    'Val_loss/PSNR_res_parts',{
                        'CT2CT':avg_val_psnr_ct2ct,
                        'CT2MR':avg_val_psnr_mr,
                        'MR2CT':avg_val_psnr_ct,
                        'MR2MR':avg_val_psnr_mr2mr
                    }, epoch_nums
                )
                
                writer.add_scalars(
                    'Val_loss/perceptual_parts',{
                        'CT2MR':avg_val_percetual_ct2mr,
                        'MR2CT':avg_val_percetual_mr2ct,
                    }, epoch_nums
                )
                
                writer.add_scalars(
                    'Val_loss/LPIPS_parts',{
                        'CT2MR':avg_val_lpips_ct2mr,
                        'MR2CT':avg_val_lpips_mr2ct,
                    }, epoch_nums
                )
            if avg_val_psnr_ct>=best_mr2ct_psnr:
                best_mr2ct_psnr = avg_val_psnr_ct
                if dist.get_rank() == 0:
                    torch.save(generator.module.state_dict(), os.path.join(model_save_path, 'best_generator.pth'))
                    torch.save(disc_ct.module.state_dict(), os.path.join(model_save_path, 'best_discriminator_delay.pth'))
                    torch.save(disc_mr.module.state_dict(), os.path.join(model_save_path, 'best_discriminator_pre.pth'))
            
            
            if epoch_nums % model_save_interval == 0 and epoch_nums != 0 and dist.get_rank() == 0:
                print('epoch:', epoch_nums, 'has been saved')
                torch.save(generator.module.state_dict(), os.path.join(model_save_path, 'generator_%d.pth'%epoch_nums))
                torch.save(disc_ct.module.state_dict(), os.path.join(model_save_path, 'discriminator_delay_%d.pth'%epoch_nums))
                torch.save(disc_mr.module.state_dict(), os.path.join(model_save_path, 'discriminator_pre_%d.pth'%epoch_nums))

        if dist.is_initialized():
                dist.destroy_process_group() 

    
    except KeyboardInterrupt:
        if epoch_nums >= 10 and dist.get_rank() == 0:
            print('Training has been stopped at epoch:', epoch_nums)
            print('Saving model')
            torch.save(generator.module.state_dict(), os.path.join(model_save_path, 'generator_%d.pth'%epoch_nums))
            torch.save(disc_ct.module.state_dict(), os.path.join(model_save_path, 'discriminator_delay_%d.pth'%epoch_nums))
            torch.save(disc_mr.module.state_dict(), os.path.join(model_save_path, 'discriminator_pre_%d.pth'%epoch_nums))
            if dist.is_initialized():
                dist.destroy_process_group()
            
        else:
            if dist.get_rank() == 0:
                print('Training has been stopped at epoch:', epoch_nums)
            if dist.is_initialized():
                dist.destroy_process_group() 

    
if __name__ == '__main__':

    
    device = setup_distributed()
    

    lr = 0.0001
    batch_size = 6
    num_workers = 64
    phase_1 = "c_v"
    phase_2 = "delay"
    
    
    
    model_save_path = './model_weight/0120_16chl_'+ phase_1 + '2' + phase_2 + '/'
    os.makedirs(model_save_path, exist_ok=True)

    data_path = '/root/autodl-tmp/LLD_MRI_Dataset/2d_mri_body_dataset_mutil_phase_v2'
    lesion_patient_file = '/root/autodl-tmp/LLD_MRI_Dataset/lesion_patient_list.txt'
    
    if dist.get_rank() == 0:
        writer = SummaryWriter()
        
    else:
        writer = None
    
    
    
    # 假设我们的比例是80%的数据用于训练
    val_ratio = 0.2
    image_size = (256, 256)
    random_seed = 42
    
    train_dataset= MultiModalDataset(data_path, lesion_patient_file, mode='paired', split='train', val_ratio=0.2, 
                                    image_size=image_size, random_seed=random_seed, phase_1 = phase_1, phase_2= phase_2)
    
    
    val_dataset = MultiModalDataset(data_path, lesion_patient_file, mode='paired', split='val', val_ratio=0.2, 
                                    image_size=image_size, random_seed=random_seed, phase_1 = phase_1, phase_2 = phase_2)
    
    train_sampler = DSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, 
                            batch_size = batch_size, 
                            sampler = train_sampler,
                            shuffle = False,
                            num_workers =num_workers,
                            pin_memory = True,
                            drop_last=True
                            )
    val_sampler = DSampler(val_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            sampler=val_sampler,
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory = True
                            )


    
    
    adv_loss = PatchAdversarialLoss(criterion="bce")
    lpips = LPIPS(net='alex')
    lpips = lpips.to(device)
    
    generator = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    latent_channels=16,
    num_channels=[64, 128, 256],
    num_res_blocks=2,
    norm_num_groups=32,
    norm_eps=1e-6,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=True,
    with_decoder_nonlocal_attn=True,
    use_convtranspose = True,
    use_checkpointing = False,
    use_flash_attention = True
    ).to(device)
    
    """
    Increasing the values of the latent_channels (e.g., to 32 and 64) can lead to better synthesis results, 
    but we used a minimum value of 16 in our paper.
    """
    
    
    disc_ct = PatchDiscriminator(
    spatial_dims=2,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    norm="INSTANCE"
    ).to(device)
    
    
    disc_mr = PatchDiscriminator(
    spatial_dims=2,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    norm="INSTANCE"
    ).to(device)
    
    perceptual_loss = PerceptualLoss(
    spatial_dims=2,
    network_type="resnet50",
    is_fake_3d=True,
    fake_3d_ratio=0.2,
    pretrained=False,
    pretrained_path=None,
    pretrained_state_dict_key="state_dict"
    ).to(device)
    
    load_checkpoint = False

    if load_checkpoint:
        generator.load_state_dict(torch.load('model_weight/0925_2d_mutil_gpu_all_body/generator_150.pth', map_location=torch.device(device), weights_only=True))
        
        disc_ct.load_state_dict(torch.load('model_weight/0925_2d_mutil_gpu_all_body/discriminator_delay_150.pth', map_location=torch.device(device), weights_only=True))
        
        disc_mr.load_state_dict(torch.load('model_weight/0925_2d_mutil_gpu_all_body/discriminator_pre_150.pth', map_location=torch.device(device), weights_only=True))
        
        print(f"Rank {dist.get_rank()} load succeed.")

    

    
    generator = DDP(generator, device_ids=[device], find_unused_parameters=True)
    disc_ct = DDP(disc_ct, device_ids = [device], find_unused_parameters=False)
    disc_mr = DDP(disc_mr, device_ids = [device], find_unused_parameters=False)
    

    
    optimizer_gen = optim.Adam(
        generator.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )
    optimizer_disc_ct = optim.Adam(
        disc_ct.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )
    
    optimizer_disc_mr = optim.Adam(
        disc_mr.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )

    def lr_lambda(epoch):
        if epoch < 15:
            return 1.0
        # elif epoch < 30:
        #     return 0.5
        else:
            return 1.0

    scheduler_gen = optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lr_lambda)
    scheduler_disc_ct = optim.lr_scheduler.LambdaLR(optimizer_disc_ct, lr_lambda=lr_lambda)
    scheduler_disc_mr = optim.lr_scheduler.LambdaLR(optimizer_disc_mr, lr_lambda=lr_lambda)

    train_VAEYnet(
        train_loader = train_loader,
        train_sampler = train_sampler,
        val_sampler = val_sampler,
        val_loader = val_loader,
        generator=generator,
        disc_ct = disc_ct,
        disc_mr = disc_mr,
        opt_gen = optimizer_gen,
        opt_ct = optimizer_disc_ct,
        opt_mr = optimizer_disc_mr,
        device = device,
        writer=writer,
        scheduler_gen = scheduler_gen,
        scheduler_disc_ct = scheduler_disc_ct,
        scheduler_disc_mr = scheduler_disc_mr,
        perceptual_loss = perceptual_loss,
        model_save_interval=10,
        model_save_path = model_save_path,
        adv_loss = adv_loss,
        lpips=lpips
    )