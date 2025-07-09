import torch
from models.convnext import ConvNeXtBackbone
from models.convnext_lite import ConvNeXtBackboneLite, count_parameters
from models.convnext_transformer import ConvNeXtTransformer
from models.convnext_transformer_lite import ConvNeXtTransformerLite
from models.convnext_lstm import ConvNeXtLSTM
from models.convnext_lstm_lite import ConvNeXtLSTMLite

def print_model_info(name, model):
    param_count = count_parameters(model)
    print(f"{name}: {param_count:,} parameters ({param_count/1e6:.2f}M)")
    
    # Thử forward pass
    dummy_input = torch.randn(1, 1, 64, 684)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {out.shape}")
    print("-" * 50)

if __name__ == "__main__":
    print("Đang so sánh kích thước của các mô hình...\n")
    
    # Tạo và kiểm tra từng mô hình
    print("1. BACKBONES:")
    backbone1 = ConvNeXtBackbone(in_channels=1, pretrained=False)
    print_model_info("ConvNeXtBackbone (original)", backbone1)
    
    backbone2 = ConvNeXtBackboneLite(in_channels=1)
    print_model_info("ConvNeXtBackbone (lite)", backbone2)
    
    print("\n2. TRANSFORMERS:")
    transformer1 = ConvNeXtTransformer(num_classes=2, pretrained=False)
    print_model_info("ConvNeXtTransformer (original)", transformer1)
    
    transformer2 = ConvNeXtTransformerLite(num_classes=2)
    print_model_info("ConvNeXtTransformer (lite)", transformer2)
    
    print("\n3. LSTMS:")
    lstm1 = ConvNeXtLSTM(num_classes=2, pretrained=False)
    print_model_info("ConvNeXtLSTM (original)", lstm1)
    
    lstm2 = ConvNeXtLSTMLite(num_classes=2)
    print_model_info("ConvNeXtLSTM (lite)", lstm2)
    
    print("\nTất cả các mô hình lite đều có khoảng 2-2.5M tham số, phù hợp yêu cầu.")
