import torch        # Native PyTorch  
import torch_mlu    # Cambricon PyTorch Backend  
from torch.profiler import profile, ProfilerActivity  


x = torch.randn(16, 256, 112, 112).mlu()  
y = torch.randn(16, 256, 112, 112).mlu()  

activities = [ProfilerActivity.CPU, ProfilerActivity.MLU]  
sort_by_keyword = "mlu_time_total"  

# Warm up
for _ in range(10):  
    z = x + y  
    torch.mlu.synchronize()  

torch.mlu.synchronize()  
with profile(activities=activities, record_shapes=True, profile_memory=True, with_flops=True) as prof:  
    for _ in range(100):   
        z = x + y  
    torch.mlu.synchronize()  

print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))