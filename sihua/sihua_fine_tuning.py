import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from genslm import GenSLM, SequenceDataset
from Bio import SeqIO
import torch.optim as optim

# 假设模型和数据集已经按照您的要求加载和准备好
model = GenSLM("genslm_2.5b_patric", model_cache_dir="/scratch/sp96859/GenSLM")
model.train()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sequences = [str(record.seq) for record in SeqIO.parse("sihua.fasta", "fasta")]
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)

# 分割数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    total_train_loss = 0
    model.train()
    for batch in train_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss  # 假设模型输出包含损失
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    # 验证循环
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss  # 同样假设模型输出包含损失
            total_test_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Train Loss: {total_train_loss / len(train_dataloader)}, Test Loss: {total_test_loss / len(test_dataloader)}")

# 可选：保存微调后的模型
torch.save(model.state_dict(), "sihua_2.5b_model.pt")
