import wandb
from pytorch_lightning.utilities import rank_zero_only


# TODO: Check if it is necessary (for multi-GPU training) to use the PyTorchLightning logging wrapper to log the table


class ReconstructionLogger:
    def __init__(
        self,
        logging_interval,
        samples_per_train_interval,
        samples_per_val_interval,
        num_samples_per_z_to_consider,
    ):
        self.logging_interval = logging_interval
        self.samples_per_train_interval = samples_per_train_interval
        self.samples_per_val_interval = samples_per_val_interval
        self._last_logged_step = -1
        self.train_counter = 0
        self.val_counter = 0
        self.num_samples_per_z_to_consider = num_samples_per_z_to_consider

    def num_samples_to_log(self, trainer, train):
        if trainer.sanity_checking or trainer.fast_dev_run:
            return 0

        if train:
            return self._samples_to_log_train(trainer)

        return self._samples_to_log_val(trainer)

    def _samples_to_log_train(self, trainer):
        if trainer.global_step % self.logging_interval == 0 and trainer.global_step != self._last_logged_step:
            self._last_logged_step = trainer.global_step
            self.train_counter = self.samples_per_train_interval

        return self.train_counter

    def _samples_to_log_val(self, trainer):
        return self.val_counter

    def reset_val_counter(self):
        self.val_counter = self.samples_per_val_interval

    @rank_zero_only
    def log_table(
        self, table_name, train, row_dicts=None, columns=None, row_list=None, additional_to_log={}, trainer=None
    ):
        """
        The functions supports two interfaces. One with dictionaries (default) and one with columns and data lists.

        Interface 1:
        row_dicts: a list of dictionaries where each dictionary corresponds to a row in the table
        The set of keys must be shared across all the dictionaries.

        Interface 2:
        columns: a list of column names
        data: a list of lists where each list corresponds to a row in the table

        Common parameters:
        table_name: The name which would be use to log the table
        train: whether this is log during training
        additional_to_log: (optional) A dictionary of key:value pairs that will be logged with the table
        trainer: (optional) Could be used for logging using the PyTorchLightning wrapper
        """
        assert row_dicts or (columns and row_list)

        if row_dicts is None:
            table = wandb.Table(columns=columns)
            for row_data in row_list:
                table.add_data(*row_data)
            num_logged_samples = len(row_list)
        else:
            columns = row_dicts[0].keys
            table = wandb.Table(columns=columns)
            for row_dict in row_dicts:
                row_data = [row_dict[col_name] for col_name in columns]
                table.add_data(*row_data)
            num_logged_samples = len(row_dicts)

        if train:
            self.train_counter -= num_logged_samples
        else:
            self.val_counter -= num_logged_samples

        additional_to_log[table_name] = table
        wandb.log(additional_to_log)
