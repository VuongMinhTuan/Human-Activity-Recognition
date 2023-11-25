from torch.optim.lr_scheduler import SequentialLR , LambdaLR
from torch.optim.optimizer import Optimizer
from bisect import bisect_right



class SequentialLR(SequentialLR):
    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        scheduler.step()
        self._last_lr = scheduler.get_last_lr()


def scheduler_with_warmup(
        scheduler: Optimizer,
        warmup_epochs: int,
        start_factor: float = 0.1
):
    
    warmup_scheduler = LambdaLR(scheduler.optimizer, lr_lambda= lambda epoch: start_factor + (1 - start_factor) * (epoch / warmup_epochs))
    
    return SequentialLR(
        optimizer = scheduler.optimizer,
        schedulers = [warmup_scheduler, scheduler],
        milestones = [warmup_epochs+1]
    )