import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.data import DataLoader
from dataloader import train_ds, test_ds

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def prepare_dataset():
    train_loader = DataLoader(dataset = train_ds, batch_size=2, shuffle=False, pin_memory=True, drop_last=True, sampler=DistributedSampler(train_ds))
    test_loader = DataLoader(dataset = test_ds, batch_size=2, shuffle=False, pin_memory=True, drop_last=True, sampler=DistributedSampler(test_ds))
    return train_loader, test_loader

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()

    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            ## 모델 예측과 역전파 코드 추가
            print(f"에포크: {epoch+1:03d}/{num_epochs:03d}"
                f" | 배치 크기 {labels.shape[0]:03d}"
                f" | 훈련 손실: {loss:.2f}")
        
        model.eval()
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] 훈련 정확도", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] 테스트 정확도", test_acc)
        destroy_process_group()

    if __name__ == "__main__":
        print("사용 가능한 GPU 개수:", torch.cuda.device_count())
        torch.manual_seed(123)
        num_epochs = 3
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)

        