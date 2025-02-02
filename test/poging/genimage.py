import torch
from torchvision.utils import save_image

# 加载预训练的GAN生成器
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512', pretrained=True)

# 生成随机噪声
noise = torch.randn(1, 512)  # 512维的潜在空间

# 使用模型的test方法生成图像
with torch.no_grad():
    generated_image = model.test(noise)

# 保存生成的图像
save_image(generated_image, 'generated_image.png')
