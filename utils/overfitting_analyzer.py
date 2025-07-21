import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_training_progress(history_file="training_history_anti_overfit.csv"):
    """
    Phân tích quá trình training để phát hiện overfitting
    """
    try:
        df = pd.read_csv(history_file)
    except FileNotFoundError:
        print(f"❌ File {history_file} không tồn tại!")
        return
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Analysis - Overfitting Detection', fontsize=16)
    
    # 1. Loss curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1 Score curves
    axes[0, 1].plot(df['epoch'], df['train_f1'], label='Train F1', color='green', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_f1'], label='Validation F1', color='orange', linewidth=2)
    axes[0, 1].set_title('F1 Score Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Overfitting Gap
    axes[1, 0].plot(df['epoch'], df['overfitting_gap'], color='purple', linewidth=2)
    axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Warning Threshold (0.1)')
    axes[1, 0].axhline(y=0.15, color='darkred', linestyle='--', alpha=0.7, label='Danger Threshold (0.15)')
    axes[1, 0].set_title('Overfitting Gap (Train F1 - Val F1)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gap')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance improvement
    val_f1_diff = df['val_f1'].diff().fillna(0)
    axes[1, 1].bar(df['epoch'], val_f1_diff, color=['green' if x > 0 else 'red' for x in val_f1_diff])
    axes[1, 1].set_title('Validation F1 Improvement per Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Change')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Phân tích số liệu
    print("📊 OVERFITTING ANALYSIS REPORT")
    print("=" * 50)
    
    # Best performance
    best_val_epoch = df['val_f1'].idxmax() + 1
    best_val_f1 = df['val_f1'].max()
    best_train_f1_at_best_val = df.loc[df['val_f1'].idxmax(), 'train_f1']
    
    print(f"🏆 Best Validation F1: {best_val_f1:.4f} at epoch {best_val_epoch}")
    print(f"📈 Train F1 at best val: {best_train_f1_at_best_val:.4f}")
    print(f"🔍 Gap at best val: {best_train_f1_at_best_val - best_val_f1:.4f}")
    
    # Overfitting detection
    avg_gap = df['overfitting_gap'].mean()
    max_gap = df['overfitting_gap'].max()
    final_gap = df['overfitting_gap'].iloc[-1]
    
    print(f"\n📉 Overfitting Metrics:")
    print(f"   Average gap: {avg_gap:.4f}")
    print(f"   Maximum gap: {max_gap:.4f}")
    print(f"   Final gap: {final_gap:.4f}")
    
    # Overfitting warnings
    if max_gap > 0.15:
        print("🚨 SEVERE OVERFITTING DETECTED!")
    elif max_gap > 0.1:
        print("⚠️  Moderate overfitting detected")
    else:
        print("✅ Overfitting under control")
    
    # Stability analysis
    val_f1_std = df['val_f1'].std()
    recent_epochs = df.tail(5)
    recent_improvement = recent_epochs['val_f1'].max() - recent_epochs['val_f1'].min()
    
    print(f"\n📊 Training Stability:")
    print(f"   Val F1 std: {val_f1_std:.4f}")
    print(f"   Recent improvement: {recent_improvement:.4f}")
    
    if val_f1_std > 0.05:
        print("⚠️  High validation variance - consider more regularization")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if final_gap > 0.1:
        print("   - Increase dropout rate")
        print("   - Add more data augmentation")
        print("   - Reduce model complexity")
        print("   - Increase weight decay")
    
    if recent_improvement < 0.01:
        print("   - Consider early stopping")
        print("   - Learning rate may be too low")
    
    if df['val_loss'].is_monotonic_increasing:
        print("   - Validation loss increasing - clear overfitting")
        print("   - Stop training and use earlier checkpoint")

def compare_models(files_and_names):
    """
    So sánh nhiều file training history
    """
    plt.figure(figsize=(12, 8))
    
    for file_path, name in files_and_names:
        try:
            df = pd.read_csv(file_path)
            plt.plot(df['epoch'], df['val_f1'], label=f'{name} (Val F1)', linewidth=2)
            plt.plot(df['epoch'], df['overfitting_gap'], '--', label=f'{name} (Gap)', alpha=0.7)
        except FileNotFoundError:
            print(f"⚠️ File {file_path} không tìm thấy")
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Model Comparison - Validation F1 and Overfitting Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Chạy phân tích
    analyze_training_progress()
    
    # Ví dụ so sánh models (uncomment nếu có nhiều file)
    # compare_models([
    #     ("training_history_anti_overfit.csv", "Anti-Overfit"),
    #     ("training_history_original.csv", "Original")
    # ])
