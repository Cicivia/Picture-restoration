from flask import Flask, request, send_file, make_response
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageChops
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import re
import cv2


app = Flask(__name__)
CORS(app)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 图像矩阵由 hwc 转换为 chw ，这个就不多解释了
def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])
# 图像矩阵由 chw 转换为 hwc ，这个也不多解释


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # 3 ==> 32 的输入卷积
        self.inc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True))

        # 32 ==> 32 的中间卷积
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 32 ==> 3 的输出卷积
        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 第 1 次卷积
        conv1 = self.inc(x)
        # 第 2 次卷积
        conv2 = self.conv(conv1)
        # 第 3 次卷积
        conv3 = self.conv(conv2)
        # 第 4 次卷积
        conv4 = self.conv(conv3)
        # 第 5 次卷积
        conv5 = self.outc(conv4)
        return conv5


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    # forward 需要两个输入，x1 是需要上采样的小尺寸 feature map
    # x2 是以前的大尺寸 feature map，因为中间的 pooling 可能损失了边缘像素，
    # 所以上采样以后的 x1 可能会比 x2 尺寸小
    def forward(self, x1, x2):
        # x1 上采样
        x1 = self.up(x1)

        # 输入数据是四维的，第一个维度是样本数，剩下的三个维度是 CHW
        # 所以 Y 方向上的悄寸差别在 [2],  X 方向上的尺寸差别在 [3]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 给 x1 进行 padding 操作
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        # 把 x2 加到反卷积后的 feature map
        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(6, 64),
            single_conv(64, 64))

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128))

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256))

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128))

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64))

        self.outc = outconv(64, 3)

    def forward(self, x):
        # input conv : 6 ==> 64 ==> 64
        inx = self.inc(x)

        # 均值 pooling, 然后 conv1 : 64 ==> 128 ==> 128 ==> 128
        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        # 均值 pooling，然后 conv2 : 128 ==> 256 ==> 256 ==> 256 ==> 256 ==> 256 ==> 256
        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        # up1 : conv2 反卷积，和 conv1 的结果相加，输入256，输出128
        up1 = self.up1(conv2, conv1)
        # conv3 : 128 ==> 128 ==> 128 ==> 128
        conv3 = self.conv3(up1)

        # up2 : conv3 反卷积，和 input conv 的结果相加，输入128，输出64
        up2 = self.up2(conv3, inx)
        # conv4 : 64 ==> 64 ==> 64
        conv4 = self.conv4(up2)

        # output conv: 65 ==> 3，用1x1的卷积降维，得到降噪结果
        out = self.outc(conv4)
        return out


def denoise_image(noisy_img):
    # 使用OpenCV的fastNlMeansDenoisingColored函数进行降噪
    # 参数说明：
    # src: 输入的彩色图像
    # dst: 输出的降噪后的图像
    # h: 决定过滤器强度，h值高可以很好地去除噪声，但也会把图像的细节抹去
    # hForColorComponents: 与h相同，但用于彩色图像
    # templateWindowSize: 奇数，用于计算平均值
    # searchWindowSize: 奇数，用于搜索相似的窗口
    denoised_img = cv2.fastNlMeansDenoisingColored(noisy_img, None, 25, 25, 9, 27)
    # 返回降噪后的图像
    denoised_img = adjust_contrast(denoised_img)

    return denoised_img


def clearn(img):



    img_pil = Image.fromarray(img)  # 将OpenCV的图片转换为PIL的图片
    contrast = ImageEnhance.Contrast(img_pil)  # 创建对比度对象
    img_contrast = contrast.enhance(1.5)  # 增强对比度
    sharpness = ImageEnhance.Sharpness(img_contrast)  # 创建锐度对象
    img_sharp = sharpness.enhance(1.5)  # 增强锐度
    img_sharp = np.array(img_sharp)
    img_sharp = adjust_contrast(img_sharp)
    return img_sharp

class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()

    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return noise_level, out


def load_model():
    global the_model
    the_model = torch.load("D:\\IDEA\\.IntelliJIdea\\Picture-restoration\\CBDNet\\CBDNet_Model.pth", map_location=torch.device('cpu'))
    the_model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# for ind, test_img_path in enumerate(test_fns):
load_model()


def adjust_contrast(img):
    inverted_img = 255-img # 反转图像颜色
    # inverted_img = swap_yellow_and_blue(inverted_img)
    #inverted_img = sharpen_image_with_opencv(inverted_img)
    return inverted_img


def swap_yellow_and_blue(img):
    # 假设图像是uint8类型，每个像素值在0到255之间
    swapped_img = np.flip(img, axis=2) # 沿着第三个维度反转数组
    return swapped_img


def sharpen_image_with_opencv(img):
    # 创建一个锐化卷积核
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # 使用filter2D函数将卷积核应用到图像上
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def process_image(imag):
    with torch.no_grad():

        noisy_img = imag
        noisy_img = noisy_img / 255.0
        noisy_img = np.array(noisy_img).astype('float32')
        temp_noisy_img_chw = hwc_to_chw(noisy_img)
        input_var = torch.from_numpy(temp_noisy_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0).to(device)
        _, output = the_model(input_var)
        output_np = output.squeeze().cpu().detach().numpy()
        output_np = chw_to_hwc(np.clip(output_np, 0, 1))
        return output_np


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])


@app.route('/upload', methods=['POST'])
def upload_file():

    # 获取上传的文件
    file = request.files['file']

    # 检查文件是否有文件名
    if file is None or file.filename == '':
        return 'No selected file'

    # 打开上传的图像文件
    img = Image.open(file)

    # 获取图片的通道数
    n_channel = len(img.split())

    # 如果图片是四通道的，去掉Alpha通道，只保留RGB通道
    if n_channel == 4:
        img = img.convert('RGB')
    noisy_img = np.array(img)
    # noisy_img = noisy_img[:, :, ::-1]

    #Clearn函数


    #函数降噪
    # Target = denoise_image(noisy_img)



    #使用CBDNet模型降噪
    Target = process_image(noisy_img)

    result_dir = 'D:\\IDEA\\.IntelliJIdea\\Picture-restoration\\web_app\\src\\assets\\test\\'
    Image.fromarray(np.uint8(Target)).save(fp=result_dir + 'test000.jpg', format='JPEG')
    Target = Image.fromarray((Target * 255).astype(np.uint8))

    response = BytesIO()
    Target.save(response, format='JPEG')
    response.seek(0)

# 构建Flask响应对象
    response = make_response(send_file(response, mimetype='image/jpeg'))

    # 允许跨域请求
    response.headers['Access-Control-Allow-Origin', 'Content-Type'] = '*'
    print('ok')
    return response



if __name__ == '__main__':
    app.run(debug=True)
