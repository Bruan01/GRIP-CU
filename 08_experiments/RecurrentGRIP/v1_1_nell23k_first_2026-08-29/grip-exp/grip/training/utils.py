from transformers import TrainerCallback


class EarlyStoppingOnTrainLossCallback(TrainerCallback):
    """
    A callback that stops training when the training loss falls below a specified threshold.
    """
    def __init__(self, threshold: float = 0.4, min_epoch: int = 1):
        self.threshold = threshold
        self.min_epoch = min_epoch

    def on_log(self, args, state, control, logs=None, **kwargs):
        #TODO: determine by both loss and train epoch.

        # logs contains metrics like 'loss', 'learning_rate', etc.
        if logs is None or "loss" not in logs:
            return

        current_loss = logs["loss"]
        epoch = logs["epoch"]
        if current_loss < self.threshold and epoch >= self.min_epoch:
            control.should_training_stop = True
            print(f"\nStopping early: training loss {current_loss:.4f} < threshold {self.threshold:.4f} "
                  f"or training epoch {epoch} >= min_epoch {self.min_epoch}.\n")