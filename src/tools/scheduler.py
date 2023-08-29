import math


class CosineSchedule:
    def __init__(self, min_lr, init_lr, decay_rate, max_epochs) -> None:
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.max_epochs = max_epochs

    def __call__(self, optimizer, epoch):
        """Decay the learning rate"""
        lr = (self.init_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * epoch / self.max_epochs)
        ) + self.min_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class StepSchedule:
    def __init__(self, min_lr, init_lr, decay_rate) -> None:
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.decay_rate = decay_rate

    def __call__(self, optimizer, epoch):
        lr = max(self.min_lr, self.init_lr * (self.decay_rate**epoch))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
