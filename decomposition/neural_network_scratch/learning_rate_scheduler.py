class LearningRateScheduler:
    
    
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step(self, epoch: int) -> float:
        
        raise NotImplementedError
    
    def get_lr(self) -> float:
        
        return self.current_lr


class StepLR(LearningRateScheduler):
    
    
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self, epoch: int) -> float:
        
        if epoch > 0 and epoch % self.step_size == 0:
            self.current_lr *= self.gamma
        return self.current_lr


class ExponentialLR(LearningRateScheduler):
    
    
    def __init__(self, initial_lr: float, gamma: float = 0.95):
        super().__init__(initial_lr)
        self.gamma = gamma
    
    def step(self, epoch: int) -> float:
        #Exponential decay: lr = initial_lr * gamma^epoch.
        self.current_lr = self.initial_lr * (self.gamma ** epoch)
        return self.current_lr
