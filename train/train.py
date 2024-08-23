import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
from dataloader import get_loaders
import os
from tqdm import tqdm
from mask_model.model import MaskGenerator
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from torch import autocast, inference_mode
from prompt_to_prompt.ptp_classes import AttentionStore, prepare_unet

class Mask_Model_Pipline():
    def __init__(self, config, device="cuda", load_path=None):
        cuda_availdabe = torch.cuda.is_available()
        if cuda_availdabe and device != "cpu":
            print('Initializing model on GPU')
        else:
            print('Initializing model on CPU')

        self.device = device
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.cfg_src = config["cfg_src"]
        self.eta =  config["eta"]# = 1
        self.cfg_scale_tar_list = config["cfg_scale_tar_list"]
        self.num_diffusion_steps = config["num_diffusion_steps"]
        
        self.model_id = "/data/worker/Resources/34/HuggingFaceRepos/CompVis/stable-diffusion-v1-4"
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(self.model_id).to(device) 
        
        # 冻结 Stable Diffusion 模型的参数
        self.ldm_stable.unet.requires_grad_(False)
        self.ldm_stable.vae.requires_grad_(False)
        self.ldm_stable.text_encoder.requires_grad_(False)

        self.mask_model = MaskGenerator()

        if cuda_availdabe and device != "cpu":
            self.mask_model.cuda()

        if load_path is not None:
            self.mask_model.load_state_dict(load_path)

    def train_or_eval(self, dataloader, optimizer=None, train=False):
        losses = []

        assert not train or optimizer != None
        if train:
            self.mask_model.train()
        else:
            self.mask_model.eval()

        num_batches = len(dataloader)

        for data in dataloader:
            if train:
                optimizer.zero_grad()
                
            images, prompt_src = data
            images = images.to(self.device)

            self.forward(images, prompt_src)
            loss = self.inverse(self.xt, self.zs, self.xts, prompt_src)

            losses.append(loss.item())

            if train:
                loss.backward()
                optimizer.step()

        avg_loss = round(np.sum(losses), 4)
        avg_loss /= num_batches

        return avg_loss

    
    def forward(self, images, prompt_src):
        cfg_scale_src = self.cfg_src
        eta = self.eta # = 1

        self.ldm_stable.scheduler = DDIMScheduler.from_config(self.model_id, subfolder = "scheduler")    
        self.ldm_stable.scheduler.set_timesteps(self.num_diffusion_steps)

        x0 = images
        # vae encode image
        with inference_mode():
            w0 = (self.ldm_stable.vae.encode(x0).latent_dist.mode() * self.ldm_stable.vae.config.scaling_factor).float()

        # find Zs and wts - forward process
        self.xt, self.zs, self.xts = self.inversion_forward_process(self.ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=False, num_inference_steps = self.num_diffusion_steps)
        
    
    def inverse(self, wt, zs, wts, prompt_tar):
        attention_store = AttentionStore(
                            average=False,
                            batch_size=self.batch_size,
                        )

        prepare_unet(self.ldm_stable, attention_store)
        
        for cfg_scale_tar in self.cfg_scale_tar_list:
            loss = self.inversion_reverse_process(self.ldm_stable, xT=wts[self.num_diffusion_steps], etas=self.eta, prompts=prompt_tar, cfg_scales=[cfg_scale_tar], prog_bar=False, zs=zs[:(self.num_diffusion_steps)], controller=attention_store)
            
        return loss
        
    
    def sample_xts_from_x0(self, model, x0, num_inference_steps=50):
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
        xts = torch.zeros((num_inference_steps+1, x0.shape[0], model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
        
        xts[0] = x0
        for t in reversed(timesteps):
            idx = num_inference_steps-t_to_idx[int(t)]
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) +  torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    
        return xts
    
    
    def encode_text(self, model, prompts):
        text_input = model.tokenizer(
            prompts,
            padding="max_length",
            max_length=model.tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
            
        print(text_encoding.shape)
        return text_encoding
    
    
    def forward_step(self, model, model_output, timestep, sample):
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
    
    
    def get_variance(self, model, timestep): #, prev_timestep):
        prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def inversion_forward_process(self, model, x0, 
                                etas = None,    
                                prog_bar = False,
                                prompt = "",
                                cfg_scale = 3.5,
                                num_inference_steps=50, eps = None):
    
        if not prompt=="":
            text_embeddings = self.encode_text(model, prompt)
        uncond_embedding = self.encode_text(model, [""] * self.batch_size)
        timesteps = model.scheduler.timesteps.to(model.device)
        variance_noise_shape = (
            num_inference_steps,
            self.batch_size,
            model.unet.in_channels, 
            model.unet.sample_size,
            model.unet.sample_size)
        if etas is None or (type(etas) in [int, float] and etas == 0):
            eta_is_zero = True
            zs = None
        else:
            eta_is_zero = False
            if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
            xts = self.sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
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
                xt = xts[idx+1]
                # xt = xts_cycle[idx+1][None]
            
            with torch.no_grad():
                out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = uncond_embedding)
                if not prompt=="":
                    cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states = text_embeddings) 
    
            if not prompt=="":
                ## classifier free guidance
                noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
            else:
                noise_pred = out.sample
            if eta_is_zero:
                # 2. compute more noisy image and set x_t -> x_t+1
                xt = self.forward_step(model, noise_pred, t, xt)
    
            else: 
                # xtm1 =  xts[idx+1][None]
                xtm1 =  xts[idx][None]
                # pred of x0
                pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5
                
                # direction to xt
                prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
                alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
                
                variance = self.get_variance(model, t)
                pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * noise_pred
    
                mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    
                z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
                zs[idx] = z
    
                # correction to avoid error accumulation
                xtm1 = mu_xt + ( etas[idx] * variance ** 0.5 )*z
                xts[idx] = xtm1
                
        if not zs is None: 
            zs[0] = torch.zeros_like(zs[0]) 
    
        return xt, zs, xts
    
    
    def reverse_step(self, model, model_output, timestep, sample, eta = 0, variance_noise=None):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # 5. compute variance: "sigma_t(��)" -> see formula (16)
        # ��_t = sqrt((1 ? ��_t?1)/(1 ? ��_t)) * sqrt(1 ? ��_t/��_t?1)    
        # variance = self.scheduler._self.get_variance(timestep, prev_timestep)
        variance = self.get_variance(model, timestep) #, prev_timestep)
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
    
    def inversion_reverse_process(self, model,
                        xT, 
                        etas = 0,
                        prompts = "",
                        cfg_scales = None,
                        prog_bar = False,
                        zs = None,
                        controller=None,
                        asyrp = False):
        cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)
    
        text_embeddings = self.encode_text(model, prompts)
        uncond_embedding = self.encode_text(model, [""] * self.batch_size)
    
        if etas is None: etas = 0
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
        assert len(etas) == model.scheduler.num_inference_steps
        timesteps = model.scheduler.timesteps.to(model.device)
    
        xt = xT.expand(self.batch_size, -1, -1, -1)
        op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:] 
    
        t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}
        
        total_loss = 0  # 初始化总损失
        tmp = 0
    
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
            
            noise_guidance_edit_tmp = cond_out.sample - uncond_out.sample
    
            resolution = xT.shape[-2:]
            att_res = (int(resolution[0] / 4), int(resolution[1] / 4))
            
            out = controller.aggregate_attention(
                attention_maps=controller.step_store,
                prompts= [prompts],
                res=att_res,
                from_where=["up", "down"],
                is_cross=True,
                select= 0
            )
            controller.between_steps(store_step=False)
    
            z = zs[idx] if not zs is None else None
            z = z.expand(self.batch_size, -1, -1, -1)
            
            x = self.xts[idx]
            if tmp != 0:
                # 计算 x 和 xt 之间的 MSE 损失
                mse_loss = F.mse_loss(xt, x)
                total_loss += mse_loss  
            else:
                tmp = 1
            
            t_input = torch.tensor([float(t)] * self.batch_size)
            t_input = t_input.to(self.device)
            mask = self.mask_model(uncond_noise=uncond_out.sample, cond_noise=cond_out.sample, upsampled_feature_map=out, t=t_input, txt_embedding=text_embeddings)
            
            if prompts:
                ## classifier free guidance
                noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample) * mask
            else: 
                noise_pred = uncond_out.sample
                
            # 2. compute less noisy image and set x_t -> x_t-1  
            xt = self.reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z) 
            if controller is not None:
                xt = controller.step_callback(xt)  
        
        # 计算 x 和 xt 之间的 MSE 损失
        mse_loss = F.mse_loss(xt, x)
        
        # 累加到总损失
        total_loss += mse_loss
            
        return total_loss
    
    
    def train(self, optimizer=None, loss_function=None):
        if optimizer is None:
            optimizer = optim.Adam(self.mask_model.parameters(), lr=self.lr)
        
        train_loader, _, test_loader = get_loaders(batch_size=self.batch_size, valid=0.0,
                                                   train=0.8, root_dir="./data/local")

        cumulative_epoch = self.load_model()

        train_losses = []
        test_losses = []
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for _ in pbar:
            train_loss = self.train_or_eval(train_loader, optimizer, True)
            test_loss = self.train_or_eval(test_loader)

            # 将训练期间的性能指标添加到列表中
            train_losses.append(train_loss)
            test_losses.append(test_loss)
    
            # 更新进度条的显示信息
            pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)

            cumulative_epoch += 1
            if cumulative_epoch % 1 == 0:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(self.mask_model.state_dict(), "./checkpoint/Mask_model-" + str(cumulative_epoch) + ".pth")

        pbar.close()

        return train_losses, test_losses

    def test(self, dataloader=None, loss_function=None):
        self.load_model()

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
        if dataloader is None:
            _, _, dataloader = get_loaders(label2id=self.label2id, batch_size=self.batch_size, valid=0.0,
                                                       train=0.0, filename=self.filename)

        test_loss = self.train_or_eval(loss_function, dataloader)

        return test_loss

    def load_model(self, load_path=None):
        return_id = 0
        if load_path is not None:
            self.mask_model.load_state_dict(torch.load(load_path))
        else:
            pth_list = os.listdir("checkpoint")
            latest_pth = None
            for pth in pth_list:
                if pth.endswith(".pth"):
                    if latest_pth is None:
                        latest_pth = pth
                    else:
                        current_id = int(pth.split("-")[-1].split(".")[0])
                        latest_id = int(latest_pth.split("-")[-1].split(".")[0])
                        if current_id > latest_id:
                            latest_pth = pth

            if latest_pth is not None:
                print("load model from checkpoint/" + latest_pth)
                self.mask_model.load_state_dict(torch.load("checkpoint/" + latest_pth))
                return_id = int(latest_pth.split("-")[-1].split(".")[0])

        return return_id
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The setting of models')
    parser.add_argument('--train', type=str, default=True, help='train or not')
    parser.add_argument('--config', type=str, default="./config/base.yaml", help='config file')
    parser.add_argument('--device', type=str, default="cuda", help='the device to train or test the model')
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    Model = Mask_Model_Pipline(config, device)

    if args.train is True:
        train_loss, test_loss = Model.train()
    else:
        print("test")
    
    
    
    
    
                
                                 
              
              
              
              
              
                    