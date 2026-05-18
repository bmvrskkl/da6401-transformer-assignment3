class NoamScheduler:
    def __init__(self,optimizer,d_model,warmup_steps):
        self.optimizer=optimizer
        self.d_model=d_model
        self.warmup_steps=warmup_steps
        self.step_num=0

    def step(self):
        self.step_num += 1

        lr=(self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

        for p in self.optimizer.param_groups:
            p["lr"]=lr

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
