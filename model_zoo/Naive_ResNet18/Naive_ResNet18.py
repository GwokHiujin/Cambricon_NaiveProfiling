import torch
import torchvision
from torchvision import transforms
import time
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import warnings
warnings.filterwarnings("ignore")

try:
    import torch_mlu
except ImportError:
    print("import torch_mlu failed!")

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


class ResNet18Tester:
    def __init__(self, device='mlu', type='dummy'):   # device='cuda'
        self.device = torch.device(device)
        self.model = self._load_model()
        if type == 'dummy': 
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else: 
            self.transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])

    def _load_model(self):
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        return model.to(self.device).eval()

    def test_with_dummy_data(self, batch_size=1, warmup=10, repeats=100):
        dummy_input = torch.rand(batch_size, 3, 224, 224).to(self.device)
        return self._measure_inference_time(dummy_input, warmup, repeats)

    def test_with_real_data(self, batch_size=24, warmup=10, repeats=100):
        infer_dataset = CIFAR100(root='/cifar100', train=False, download=True, transform=self.transform)
        infer_dataset = DataLoader(dataset=infer_dataset, batch_size=batch_size, shuffle=False, num_workers=4)     
        sample_batch = next(iter(infer_dataset))[0].to(self.device)
        return self._measure_inference_time(sample_batch, warmup, repeats)

    def _measure_inference_time(self, input_tensor, warmup, repeats):
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        else:
            torch.mlu.synchronize()

        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(repeats):
                _ = self.model(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                else: 
                    torch.mlu.synchronize()
        
        elapsed = time.time() - start_time
        return elapsed / repeats * 1000  

if __name__ == "__main__":
    tester = ResNet18Tester(device='mlu')   # device='cuda'
    
    configs = [
        {'batch_size': 1, 'desc': "One"},
        {'batch_size': 8, 'desc': "Small"},
        {'batch_size': 32, 'desc': "Medium"}
    ]

    # Dummy Data Test
    print("=== Dummy Data Test ===")
    for cfg in configs:
        avg_time = tester.test_with_dummy_data(
            batch_size=cfg['batch_size'],
            warmup=10,
            repeats=100
        )
        print(f"{cfg['desc']} | Batch Size: {cfg['batch_size']} | Avg Time: {avg_time / cfg['batch_size']:.2f}ms")

    # True Data Test
    print("\n=== True Data Test ===")
    for cfg in configs:
        avg_time = tester.test_with_real_data(
            batch_size=cfg['batch_size'],
            warmup=5,
            repeats=50
        )
        print(f"{cfg['desc']} | Batch Size: {cfg['batch_size']} | Avg Time: {avg_time / cfg['batch_size']:.2f}ms/batch")