import base64
from io import BytesIO

import lmdb
from PIL import Image
import torch


def test_imgs():
    image_ids = [286314, 141999, 183846]
    lmdb_imgs = "datapath/datasets/MUGE/lmdb/valid/imgs"
    env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
    txn_imgs = env_imgs.begin(buffers=True)

    for image_id in image_ids:
        image_b64 = txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
        img = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
        img.show()


'''use lspci | grep -i nvidia     in linux 列出所有PCI设备并筛选出GPU信息
use  nvidia-smi    in all plantform

WDDM 模式：在Windows上，GPU运行在WDDM模式，适用于图形任务。
如果你打算在这台机器上进行深度学习或计算任务，并希望完全利用GPU资源，可能需要切换到TCC模式（仅适用于特定的NVIDIA GPU，如Quadro系列）
如果你的主要需求是提高计算性能，但使用的GPU不支持TCC模式，可以考虑使用双GPU设置，一块用于图形显示（WDDM），另一块专门用于计算任务（TCC）

解决方式 pytorch官网copy命令
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia'''

# Number of GPUs per GPU worker
def count_GPUs_pytorch():

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")


def count_GPUs_tensorflow():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs available: {len(gpus)}")


if __name__ == "__main__":

    print(torch.__version__)  # 显示 PyTorch 的版本
    print(torch.version.cuda)  # 显示 PyTorch 编译时使用的 CUDA 版本
    count_GPUs_pytorch()
    # count_GPUs_tensorflow()