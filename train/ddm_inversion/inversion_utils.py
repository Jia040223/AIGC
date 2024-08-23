import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from prompt_to_prompt.ptp_classes import LeditsGaussianSmoothing
import json

def load_real_image(folder = "data/", img_name = None, idx = 0, img_size=512, device='cuda'):
    from ddm_inversion.utils import pil_to_tensor
    from PIL import Image
    from glob import glob
    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size,
                                    img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1]== 4:
        img = img[:,:3,:,:]
    return img

def mu_tilde(model, xt,x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t 
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1-alpha_bar)) * x0 +  ((alpha_t**0.5 *(1-alpha_prod_t_prev)) / (1- alpha_bar))*xt

def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels, 
            model.unet.sample_size,
            model.unet.sample_size)
    
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps+1,model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps-t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) +  torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]


    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length, 
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding

def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                    model_output,
                                    torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep): #, prev_timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def inversion_forward_process(model, x0, 
                            etas = None,    
                            prog_bar = False,
                            prompt = "",
                            edit_threshold_c = 0.95,
                            cfg_scale = 3.5,
                            num_inference_steps=50, eps = None, attention_store=None):

    if not prompt=="":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels, 
        model.unet.sample_size,
        model.unet.sample_size)
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xt = x0
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps-t_to_idx[int(t)]-1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx+1][None]
            # xt = xts_cycle[idx+1][None]
                    
        with torch.no_grad():
            out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = uncond_embedding)
            if not prompt=="":
                cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states = text_embeddings) 

        noise_guidance_edit_tmp = cond_out.sample - out.sample
        noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
        noise_guidance_edit_tmp_quantile = torch.sum(
            noise_guidance_edit_tmp_quantile, dim=1, keepdim=True
        )
        noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(
            1, model.unet.config.in_channels, 1, 1
        )

        # torch.quantile function expects float32
        if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
            tmp = torch.quantile(
                noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                edit_threshold_c,
                dim=2,
                keepdim=False,
            )
        else:
            tmp = torch.quantile(
                noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                edit_threshold_c,
                dim=2,
                keepdim=False,
            ).to(noise_guidance_edit_tmp_quantile.dtype)

        intersect_mask = torch.where(
                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                torch.ones_like(noise_guidance_edit_tmp),
                torch.zeros_like(noise_guidance_edit_tmp),
            )

        if not prompt=="":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample) * intersect_mask
        else:
            noise_pred = out.sample
        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else: 
            # xtm1 =  xts[idx+1][None]
            xtm1 =  xts[idx][None]
            # pred of x0
            pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5
            
            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
            
            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + ( etas[idx] * variance ** 0.5 )*z
            xts[idx] = xtm1
            
        if attention_store is not None:
            attention_store.between_steps()

    if not zs is None: 
        zs[0] = torch.zeros_like(zs[0]) 

    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta = 0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) #, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z =  eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample

def inversion_reverse_process(model,
                    xT, 
                    edit_threshold_c = 0.95,
                    etas = 0,
                    prompts = "",
                    cfg_scales = None,
                    prog_bar = False,
                    zs = None,
                    controller=None,
                    asyrp = False,
                    attention_store = None):
    batch_size = len(prompts)
    mse_list = []  # 用于存储每个时间步的 MSE
    print(batch_size)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    #uncond_embedding = encode_text(model, src_prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:] 

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)    
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(xt, timestep =  t, 
                                            encoder_hidden_states = uncond_embedding)

            ## Conditional embedding  
        if prompts:  
            with torch.no_grad():
                cond_out = model.unet.forward(xt, timestep =  t, 
                                                encoder_hidden_states = text_embeddings)
        
        mse = calculate_mse(cond_out.sample[0], cond_out.sample[1])
        mse_list.append(mse)
        
        ''' 
        controller.between_steps()
        noise_guidance_edit_tmp = cond_out.sample - uncond_out.sample

        resolution = xT.shape[-2:]
        att_res = (int(resolution[0] / 4), int(resolution[1] / 4))
        
        #print(controller.attention_store[-1])
        out = controller.aggregate_attention(
            attention_maps=controller.attention_store[-1],
            prompts= [prompts],
            res=att_res,
            from_where=["up", "down"],
            is_cross=True,
            select= 0
        )
        attn_map = out[:, :, :, 1 : 2]  # 0 -> startoftext

        # average over all tokens
        attn_map = attn_map[1].unsqueeze(0)
        if attn_map.shape[3] != 1:
            raise ValueError(
                f"Incorrect shape of attention_map. Expected size 1, but found {attn_map.shape[3]}!"
            )
        attn_map = torch.sum(attn_map, dim=3)

        # gaussian_smoothing
        attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
        smoothing = LeditsGaussianSmoothing("cuda")
        attn_map = smoothing(attn_map).squeeze(1)

        # torch.quantile function expects float32
        if attn_map.dtype == torch.float32:
            tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold_c, dim=1)
        else:
            tmp = torch.quantile(
                attn_map.flatten(start_dim=1).to(torch.float32), edit_threshold_c, dim=1
            ).to(attn_map.dtype)
        attn_mask = torch.where(
            attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1, *att_res), 1.0, 0.0
        )

        # resolution must match latent space dimension
        attn_mask = F.interpolate(
            attn_mask.unsqueeze(1),
            noise_guidance_edit_tmp.shape[-2:],  # 64,64
        ).repeat(1, 4, 1, 1)      

    
        noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
        noise_guidance_edit_tmp_quantile = torch.sum(
            noise_guidance_edit_tmp_quantile, dim=1, keepdim=True
        )
        noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(
            1, model.unet.config.in_channels, 1, 1
        )

        # torch.quantile function expects float32
        if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
            tmp = torch.quantile(
                noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                edit_threshold_c,
                dim=2,
                keepdim=False,
            )
        else:
            tmp = torch.quantile(
                noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                edit_threshold_c,
                dim=2,
                keepdim=False,
            ).to(noise_guidance_edit_tmp_quantile.dtype)

        intersect_mask = torch.where(
                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                torch.ones_like(noise_guidance_edit_tmp),
                torch.zeros_like(noise_guidance_edit_tmp),
            ) * attn_mask
        '''

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z) 
        if controller is not None:
            xt = controller.step_callback(xt)    
     
    
    mse_avg = sum(mse_list) / len(mse_list) if mse_list else 0
    # 保存 MSE 列表和平均值到文件
    save_mse_to_file(mse_list, mse_avg, "tmp/mse.json")
    
    return xt, zs

def calculate_mse(tensor1, tensor2):
    mse = torch.mean((tensor1 - tensor2) ** 2)
    return mse.item()

def save_mse_to_file(mse_list, mse_avg, mse_save_path):
    # 如果文件存在，读取已有内容
    if os.path.exists(mse_save_path):
        with open(mse_save_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # 将 MSE 列表和平均值一起作为一行保存
    existing_data.append({
        "mse_list": mse_list,
        "average_mse": mse_avg
    })

    # 保存更新后的列表回文件
    with open(mse_save_path, "w") as f:
        json.dump(existing_data, f, indent=4)