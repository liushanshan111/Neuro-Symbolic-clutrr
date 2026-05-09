# utils/curriculum_trainer.py
"""
课程学习训练器
从易到难逐步训练
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np


class CurriculumTrainer:
    """
    课程学习训练器
    
    训练策略：
      阶段1 (Epoch 1-15):  只训练2跳
      阶段2 (Epoch 16-30): 训练2-3跳
      阶段3 (Epoch 31-45): 训练2-4跳
      阶段4 (Epoch 46-60): 训练所有跳数
    """
    
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 config,
                 output_dir):
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = output_dir
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器（余弦退火）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15,  # 15个epoch重启一次
            T_mult=1,
            eta_min=1e-6
        )
        
        # 损失函数
        from models.sequential_drn import SequentialRuleLoss
        self.criterion = SequentialRuleLoss()
        
        # 设备
        self.device = torch.device(
            config['device'] if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # 课程设置
        self.curriculum_stages = [
            {'epochs': (0, 15), 'max_hop': 2},
            {'epochs': (15, 30), 'max_hop': 3},
            {'epochs': (30, 45), 'max_hop': 4},
            {'epochs': (45, 60), 'max_hop': 10},
        ]
        
        # 历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_acc_by_hop': {}
        }
        
        self.best_val_acc = 0
    
    def train(self):
        """主训练循环"""
        
        print("\n" + "="*70)
        print("课程学习训练")
        print("="*70)
        
        for stage in self.curriculum_stages:
            start_epoch, end_epoch = stage['epochs']
            max_hop = stage['max_hop']
            
            print(f"\n{'='*70}")
            print(f"阶段: Epoch {start_epoch+1}-{end_epoch}")
            print(f"最大跳数: {max_hop}")
            print(f"{'='*70}")
            
            for epoch in range(start_epoch, end_epoch):
                
                # 训练
                train_metrics = self._train_epoch(epoch, max_hop)
                
                # 验证
                val_metrics = self._validate(epoch)
                
                # 记录
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                
                # 打印
                if (epoch + 1) % 3 == 0 or epoch == start_epoch:
                    print(f"\nEpoch {epoch+1}/{end_epoch}")
                    print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                          f"Acc: {train_metrics['accuracy']:.2%}")
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                          f"Acc: {val_metrics['accuracy']:.2%}")
                    print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # 保存最佳模型
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self._save_checkpoint(epoch, 'best_model_curriculum.pt')
                
                # 学习率调度
                self.scheduler.step()
        
        print(f"\n{'='*70}")
        print(f"✓ 训练完成！")
        print(f"  最佳验证准确率: {self.best_val_acc:.2%}")
        print(f"{'='*70}")
        
        return self.history
    
    def _train_epoch(self, epoch, max_hop):
        """训练一个epoch"""
        
        self.model.train()
        
        # 过滤数据
        filtered_indices = [
            i for i, sample in enumerate(self.train_dataset.samples)
            if sample['chain_length'] <= max_hop
        ]
        
        filtered_dataset = Subset(self.train_dataset, filtered_indices)
        
        train_loader = DataLoader(
            filtered_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training (max_hop={max_hop})")
        
        for batch in pbar:
            
            chains = batch['chain'].to(self.device)
            targets = batch['target'].squeeze(1).to(self.device)
            chain_lengths = batch['chain_length']
            
            # 前向
            logits, similarities, attention_weights = self.model(
                chains, 
                chain_lengths
            )
            
            # 损失
            loss, _ = self.criterion(
                logits,
                targets,
                attention_weights,
                self.model.rule_templates,
                self.model.get_relation_embeddings()
            )
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def _validate(self, epoch):
        """验证"""
        
        self.model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )
        
        total_loss = 0
        correct = 0
        total = 0
        
        # 按跳数统计
        by_hop = {}
        
        with torch.no_grad():
            for batch in val_loader:
                
                chains = batch['chain'].to(self.device)
                targets = batch['target'].squeeze(1).to(self.device)
                chain_lengths = batch['chain_length']
                
                logits, similarities, attention_weights = self.model(
                    chains,
                    chain_lengths
                )
                
                loss, _ = self.criterion(
                    logits,
                    targets,
                    attention_weights,
                    self.model.rule_templates
                )
                
                total_loss += loss.item()
                
                _, predicted = logits.max(1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                # 按跳数统计
                for i, length in enumerate(chain_lengths):
                    hop = length.item()
                    if hop not in by_hop:
                        by_hop[hop] = {'correct': 0, 'total': 0}
                    by_hop[hop]['total'] += 1
                    if predicted[i] == targets[i]:
                        by_hop[hop]['correct'] += 1
        
        # 计算按跳数准确率
        hop_accuracies = {}
        for hop in sorted(by_hop.keys()):
            acc = by_hop[hop]['correct'] / by_hop[hop]['total']
            hop_accuracies[hop] = acc
        
        self.history['val_acc_by_hop'][epoch] = hop_accuracies
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def _save_checkpoint(self, epoch, filename):
        """保存检查点"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        torch.save(checkpoint, self.output_dir / filename)