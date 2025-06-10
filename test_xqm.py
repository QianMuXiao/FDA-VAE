import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



from mri_dataset_v2 import MultiModalDataset
from Yshape_AutoencoderKL import AutoencoderKL


from lpips import LPIPS
import pandas as pd
import time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np


if __name__ == '__main__':
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 1
    num_workers = 16
    
    phase_1 = "c_v"
    phase_2 = "delay"
    
    lpips = LPIPS(net='alex').to('cuda')
    
    task_name = phase_1 + "2" + phase_2
    
    weight_path = './model_weight/' + task_name + '/' + 'generator_40.pth' 
    

    
    save_path = './results'
    
    data_path = '/root/autodl-tmp/LLD-MRIdataset/2d_mri_body_dataset_mutil_phase_v2'
    lesion_patient_file = '/root/autodl-tmp/LLD-MRIdataset/lesion_patient_list.txt'
    
    val_ratio = 0.2
    image_size = (256, 256)
    random_seed = 42

    val_psnr = 0.0
    val_ssim = 0.0
    val_l1 = 0.0
    val_lpips = 0.0



    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    def save_image(image, path, cmap='gray'):
        plt.imsave(path, image, cmap=cmap)
    total_time = 0.0
    metrics_data = []


    val_dataset = MultiModalDataset(data_path, lesion_patient_file, 
                                    mode='paired', split='val', val_ratio=0.2, 
                                    image_size=image_size, random_seed=random_seed, 
                                    phase_1 = phase_1, phase_2 = phase_2)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory = True
                            )
    
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
    
    generator.load_state_dict(torch.load(weight_path, map_location=device,weights_only=True))
    generator.eval()

    # total_params = sum(p.numel() for p in generator.parameters())
    # print(f'Total parameters: {total_params:,}')
    # trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    # print(f'Trainable parameters: {trainable_params:,}')
    
    
    
    for i, (real_im_B, real_im_A, real_B_path, real_A_path) in enumerate(tqdm(val_loader, desc='Testing', dynamic_ncols=True, total=len(val_loader))):
        start_time = time.time()
        with torch.no_grad():
            real_im_A = real_im_A.to(device)
            real_im_B = real_im_B.to(device)
            
            z_mu_A, z_simga_A = generator.encode(real_im_A)
            z_mu_A2B, z_simga_A2B = generator.flip_distribution(z_mu_A, z_simga_A)
            z_A2B = generator.sampling(z_mu_A2B, z_simga_A2B)
            fake_im_B = generator.decode_ct(z_A2B)
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            
            real_A_path = real_A_path[0]
            path_A_parts = real_A_path.split('/')
            phase_A_name = path_A_parts[-2]  # 获取倒数第二个元素 (pre)
            file_A_name = path_A_parts[-1]   # 获取最后一个元素 (MR178631_9.nii.gz)
            
            real_B_path = real_B_path[0]
            path_B_parts = real_B_path.split('/')
            phase_B_name = path_B_parts[-2]  # 获取倒数第二个元素 (pre)
            file_B_name = path_B_parts[-1]   # 获取最后一个元素 (MR178631_9.nii.gz)
            
            file_name_parts = file_A_name.split('.')[0]
        

        
            fake_B_tensor = fake_im_B
            real_B_tensor = real_im_B
            
            fake_im_B=fake_im_B.cpu().data.numpy().squeeze()
            real_im_B=real_im_B.cpu().data.numpy().squeeze()
            real_im_A=real_im_A.cpu().data.numpy().squeeze()
            diff_im = np.abs(real_im_B - fake_im_B)
            
            
            slices_l1 = np.mean(np.abs(real_im_B - fake_im_B))
            slices_lpips = lpips(real_B_tensor, fake_B_tensor).mean()
            
            fake_im_B = fake_im_B * 0.5 + 0.5
            real_im_B = real_im_B * 0.5 + 0.5
            real_im_A = real_im_A * 0.5 + 0.5
            

            
            slices_psnr = psnr(real_im_B, fake_im_B, data_range=1)
            slices_ssim = ssim(real_im_B, fake_im_B, data_range=1)

        
        metrics_data.append({
        'file_name': file_name_parts,
        'PSNR': slices_psnr,
        'SSIM': slices_ssim,
        'L1': slices_l1,
        'LPIPS': slices_lpips,
        'Inference_Time': inference_time
        })
        
        
        output_path_A = os.path.join(save_path, phase_A_name + '2' + phase_B_name, phase_A_name)
        output_path_B = os.path.join(save_path, phase_A_name + '2' + phase_B_name, phase_B_name)
        output_path_fake_B = os.path.join(save_path, phase_A_name + '2' + phase_B_name, 'fake' + phase_B_name)
        out_put_path_diff = os.path.join(save_path, phase_A_name + '2' + phase_B_name, 'diff' + phase_B_name)
        
        if not os.path.exists(output_path_A):
            os.makedirs(output_path_A)
        if not os.path.exists(output_path_B):
            os.makedirs(output_path_B)
        if not os.path.exists(output_path_fake_B):
            os.makedirs(output_path_fake_B)
        if not os.path.exists(out_put_path_diff):
            os.makedirs(out_put_path_diff)
        



        # Save real_A, real_B, fake_B, and diff_im as grayscale images
        save_image(real_im_A, os.path.join(output_path_A, file_name_parts + '_gray.png'), cmap='gray')
        save_image(real_im_B, os.path.join(output_path_B, file_name_parts + '_gray.png'), cmap='gray')
        save_image(fake_im_B, os.path.join(output_path_fake_B, file_name_parts + '_gray.png'), cmap='gray')
        save_image(diff_im, os.path.join(out_put_path_diff, file_name_parts + '_heatmap.png'), cmap='jet')


        
        
        
        val_psnr += slices_psnr
        val_ssim += slices_ssim
        val_l1 += slices_l1
        val_lpips += slices_lpips
    
    avg_time = total_time / len(val_loader)
    print(f'Average inference time: {avg_time:.4f} seconds')

    # 创建DataFrame并保存
    df = pd.DataFrame(metrics_data)
    metrics_save_path = os.path.join(save_path, phase_A_name + '2' + phase_B_name, 'metrics.csv')
    df.to_csv(metrics_save_path, index=False)

        
    
    avg_psnr = val_psnr / len(val_loader)
    avg_ssim = val_ssim / len(val_loader)
    avg_l1 = val_l1 / len(val_loader)
    avg_lpips = val_lpips / len(val_loader)
    
    print('PSNR: %.4f, SSIM: %.4f, L1: %.4f, LPIPS: %.4f' % (avg_psnr, avg_ssim, avg_l1, avg_lpips))