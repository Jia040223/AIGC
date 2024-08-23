import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_size):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)
        self.scale = embed_size ** 0.5

    def forward(self, txt_embedding):
        # [batch_size, seq_length, embed_size]
        query = self.query_layer(txt_embedding)  # [batch_size, seq_length, embed_size]
        key = self.key_layer(txt_embedding)      # [batch_size, seq_length, embed_size]
        value = self.value_layer(txt_embedding)  # [batch_size, seq_length, embed_size]

        # 计算注意力分数
        scores = torch.bmm(query, key.transpose(1, 2)) / self.scale  # [batch_size, seq_length, seq_length]
        weights = F.softmax(scores, dim=2)  # [batch_size, seq_length, seq_length]

        # 应用注意力权重
        attended_output = torch.bmm(weights, value)  # [batch_size, seq_length, embed_size]
        return attended_output


class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.conv_transpose1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv_feature = nn.Conv2d(77, 128, kernel_size=3, stride=1, padding=1)
        
        self.time_embed = nn.Linear(1, 128)
        
        self.text_attention = Attention(embed_size=768)
        self.text_embed = nn.Linear(768, 128)
        
        self.fc = nn.Linear(128 * 4, 128)
        
        self.deconv = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(64, 4, kernel_size=3, stride=2, padding=1)

    def forward(self, uncond_noise, cond_noise, upsampled_feature_map, t, txt_embedding):
        x1 = F.relu(self.conv1(uncond_noise))
        x2 = F.relu(self.conv1(cond_noise))
        
        noise_out = F.relu(self.conv2(x2 - x1))
        
        upsampled_feature_map = F.relu(self.conv_feature(upsampled_feature_map.permute(0, 3, 1, 2)))
        upsampled_feature_map = self.conv_transpose1(upsampled_feature_map)
        feature_out = self.conv_transpose2(upsampled_feature_map)
        
        # 使用注意力机制处理 txt_embedding
        txt_embed_att = self.text_attention(txt_embedding)  # [batch_size, seq_length, embed_size]
        txt_embed_att = txt_embed_att.mean(dim=1)  # [batch_size, embed_size]
        txt_embed = F.relu(self.text_embed(txt_embed_att))  # [batch_size, 128]
        txt_embed = txt_embed.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64)
        
        t_embed = F.relu(self.time_embed(t.view(-1, 1)))
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64)
        
        combined = torch.cat([noise_out, feature_out, t_embed, txt_embed], dim=1)
        combined = F.relu(self.fc(combined.permute(0,2,3,1))).permute(0,3,1,2)
        
        mask = F.relu(self.deconv(combined))
        mask = torch.sigmoid(self.final_conv(mask))
        
        return mask


def test_mask_generator():
    batch_size = 2
    
    uncond_noise = torch.randn(batch_size, 4, 64, 64)
    cond_noise = torch.randn(batch_size, 4, 64, 64)
    upsampled_feature_map = torch.randn(batch_size, 16, 16, 77)
    t = torch.tensor([10.0, 10.0])  
    txt_embedding = torch.randn(batch_size, 77,768)  
    
    model = MaskGenerator()
    
    mask = model(uncond_noise, cond_noise, upsampled_feature_map, t, txt_embedding)
    
    print(f"Generated mask shape: {mask.shape}")

if __name__ == "__main__":
    test_mask_generator()