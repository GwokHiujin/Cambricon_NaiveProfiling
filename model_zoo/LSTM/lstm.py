import torch
import torch.nn as nn
import time

try:
    import torch_mlu
except ImportError:
    print("import torch_mlu failed!")

class LSTMNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=256, num_layers=1, device='mlu'):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True).to(device)
        
        self.fc = nn.Linear(hidden_size, 1).to(device)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

def test_inference_time(model, sequence_length=256, warmup=10, repeats=100):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, sequence_length, model.lstm.input_size).to(device)  # batch_size=1
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()  
            else: 
                torch.mlu.synchronize()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / repeats * 1000  
    return avg_time

if __name__ == "__main__":
    device = torch.device('mlu')    # torch.device('cuda')
    print(f"Using device: {device}")
    
    model = LSTMNetwork(input_size=10, hidden_size=256, device=device)
    model.eval()
    
    inference_time = test_inference_time(model)
    
    print(f"Average inference time: {inference_time:.2f}ms")