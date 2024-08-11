import argparse
import yaml
import torch
import torch.optim as optim
from torch import nn
import numpy as np
from transformers import BertForTokenClassification,BertConfig
from dataloader import get_loaders
from dataset import AutoTokenizer
import os
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score
from record import record_training


class Model_Pipline():
    def __init__(self, config, device="cuda", load_path=None, row_filename="ChineseCorpus199801.txt"):
        self.filename = row_filename

        cuda_availdabe = torch.cuda.is_available()
        if cuda_availdabe and device != "cpu":
            print('Initializing model on GPU')
        else:
            print('Initializing model on CPU')

        # 超参数
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.label2id = config["label2id"]

        bert_config = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
        self.model = BertForTokenClassification(bert_config)

        if cuda_availdabe and device != "cpu":
            self.model.cuda()

        if load_path is not None:
            self.model.load_state_dict(load_path)

    def train_or_eval(self, loss_function, dataloader, optimizer=None, train=False):
        losses, precisions, recalls, f1s = [], [], [], []

        assert not train or optimizer != None
        if train:
            self.model.train()
        else:
            self.model.eval()

        num_batches = len(dataloader)

        for data in dataloader:
            if train:
                optimizer.zero_grad()

            image_w0, prompt = [d.cuda() for d in data[:]] if torch.cuda.is_available() else data[:]

            final_out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            logits = final_out.logits
            loss = loss_function(logits.view(-1, self.model.num_labels), labels.view(-1))

            losses.append(loss.item())

            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

            predictions = torch.argmax(logits, dim=-1)

            # 将标签和预测结果添加到列表中
            true_labels = labels.tolist()
            pred_labels = predictions.tolist()

            # 计算各种评估指标
            bmseo_to_iobes = {'B': 'B', 'M': 'I', 'E': 'E', 'S': 'S', '0': 'O'}
            inverse_label_map = {v: k for k, v in self.label2id.items()}
            true_labels_str = [[inverse_label_map[label] for label in sequence] for sequence in true_labels]
            pred_labels_str = [[inverse_label_map[label] for label in sequence] for sequence in pred_labels]

            true_labels_iobes = [[bmseo_to_iobes[label] for label in sequence] for sequence in true_labels_str]
            pred_labels_iobes = [[bmseo_to_iobes[label] for label in sequence] for sequence in pred_labels_str]

            precision = precision_score(true_labels_iobes, pred_labels_iobes)
            recall = recall_score(true_labels_iobes, pred_labels_iobes)
            f1 = f1_score(true_labels_iobes, pred_labels_iobes)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        avg_loss = round(np.sum(losses), 4)
        avg_loss /= num_batches
        avg_precision = round(np.sum(precisions), 4)
        avg_precision /= num_batches
        avg_recalls = round(np.sum(recalls), 4)
        avg_recalls /= num_batches
        avg_f1 = round(np.sum(f1s), 4)
        avg_f1 /= num_batches

        return avg_loss, avg_precision, avg_recalls, avg_f1

    def train(self, optimizer=None, loss_function=None):
        if optimizer is None:
            '''
            weight_decay = 0.01
            no_decay = ["bias", "norm"]

            parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            '''

            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        train_loader, _, test_loader = get_loaders(label2id=self.label2id, batch_size=self.batch_size, valid=0.0,
                                                   train=0.8, filename=self.filename)

        cumulative_epoch = self.load_model()

        train_losses, train_precisions, train_recalls, train_f1s = [], [], [], []
        test_losses, test_precisions, test_recalls, test_f1s = [], [], [], []
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for _ in pbar:
            train_loss, train_precision, train_recall, train_f1 = self.train_or_eval(loss_function, train_loader, optimizer, True)
            test_loss, test_precision, test_recall, test_f1 = self.train_or_eval(loss_function, test_loader)

            # 将训练期间的性能指标添加到列表中
            train_losses.append(train_loss)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)
            train_f1s.append(train_f1)

            # 将测试期间的性能指标添加到列表中
            test_losses.append(test_loss)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)
            test_f1s.append(test_f1)

            # 更新进度条的显示信息
            pbar.set_postfix(train_loss=train_loss, train_precision=train_precision, train_recall=train_recall, train_f1=train_f1,
                             test_loss=test_loss, test_precision=test_precision, test_recall=test_recall, test_f1=test_f1)
            '''
            print("epoch{0} : train_loss={1}, train_precision={2}, train_recall={3}, train_f1={4}"
                  .format(cumulative_epoch, train_loss, train_precision, train_recall, train_f1))
            print("epoch{0} : test_loss={1}, test_precision={2}, test_recall={3}, test_f1={4}"
                  .format(cumulative_epoch, test_loss, test_precision, test_recall, test_f1))
            '''

            cumulative_epoch += 1
            if cumulative_epoch % 1 == 0:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(self.model.state_dict(), "./checkpoint/Bert-" + str(cumulative_epoch) + ".pth")

        pbar.close()

        return train_losses, train_precisions, train_recalls, train_f1s, \
            test_losses, test_precisions, test_recalls, test_f1s

    def test(self, dataloader=None, loss_function=None):
        self.load_model()

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
        if dataloader is None:
            _, _, dataloader = get_loaders(label2id=self.label2id, batch_size=self.batch_size, valid=0.0,
                                                       train=0.0, filename=self.filename)

        test_loss, test_precision, test_recall, test_f1 = self.train_or_eval(loss_function, dataloader)

        return test_loss, test_precision, test_recall, test_f1

    def segment(self, text):
        id2label = {v: k for k, v in self.label2id.items()}

        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        raw_texts = [text]
        encoded_inputs = tokenizer(raw_texts, max_length=512)
        input_ids = torch.tensor(encoded_inputs['input_ids']).cuda()
        token_type_ids = torch.tensor(encoded_inputs['token_type_ids']).cuda()

        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).tolist()
        predictions = [[id2label[label_id] for label_id in raw[1:-1]] for raw in predictions]

        results = []
        for text, labels in zip(raw_texts, predictions):
            words = []
            word = []
            for token, label in zip(text, labels):
                if label == 'B':
                    if word:
                        words.append(''.join(word))
                        word = []
                    word.append(token)
                elif label == 'M':
                    word.append(token)
                elif label == 'E':
                    word.append(token)
                    words.append(''.join(word))
                    word = []
                else:
                    if word:
                        words.append(''.join(word))
                        word = []
                    words.append(token)

            results.append(words)

        return results

    def load_model(self, load_path=None):
        return_id = 0
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
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
                self.model.load_state_dict(torch.load("checkpoint/" + latest_pth))
                return_id = int(latest_pth.split("-")[-1].split(".")[0])

        return return_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The setting of bert models')
    parser.add_argument('--train', type=str, default=True, help='train or not')
    parser.add_argument('--config', type=str, default="./config/base.yaml", help='config file')
    parser.add_argument('--device', type=str, default="cuda", help='the device to train or test the model')
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    Model = Model_Pipline(config, device)

    if args.train is True:
        train_loss, train_precision, train_recall, train_f1, \
        test_loss, test_precision, test_recall, test_f1 = Model.train()

        filename = 'training_results_bert.txt'
        record_training(filename, train_loss, train_precision, train_recall, train_f1,
                        test_loss, test_precision, test_recall, test_f1)
    else:
        #test_losses, test_precisions, test_recalls, test_f1s = Model.test()
        #print("test loss :{0}, test_precison:{1}, test_recalls:{2}, test_f1s:{3}".formate(test_losses, test_precisions, test_recalls, test_f1s)
        Model.load_model()
        texts = ["党中央和国务院高度重视高校毕业生等青年就业创业工作。要深入学习贯彻总书记的重要指示精神，更加突出就业优先导向，千方百计促进高校毕业生就业，确保青年就业形势总体稳定。",
                 "好久不见！今天天气真好，早饭准备吃什么呀？",
                 "我特别喜欢去北京的天安门和颐和园进行游玩",
                 "中国人为了实现自己的梦想",
                 "《原神》收入大涨，腾讯、网易、米哈游位列中国手游发行商全球收入前三"]

        for text in texts:
            results = Model.segment(text)
            print("/".join(results[0]))


def inversion_forward_process(model, x0, 
                            etas = None,    
                            prog_bar = False,
                            prompt = "",
                            cfg_scale = 3.5,
                            num_inference_steps=50, eps = None):

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

        if not prompt=="":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
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
        

    if not zs is None: 
        zs[0] = torch.zeros_like(zs[0]) 

    return xt, zs, xts

def inversion_reverse_process(model,
                    xT, 
                    etas = 0,
                    prompts = "",
                    cfg_scales = None,
                    prog_bar = False,
                    zs = None):
    batch_size = len(prompts)

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
        noise_guidance_edit_tmp = cond_out.sample - uncond_out.sample

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z)  
           
    return xt, zs