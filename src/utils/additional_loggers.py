import wandb
from einops import rearrange


def get_img_rec_table_data(ids, labels, imgs, reconstructions, step, num_samples_to_log, num_samples_per_z_to_consider):
    if reconstructions.dim() != imgs.dim():
        reconstructions = rearrange(
            reconstructions,
            "(bs z_ns) (c h w) -> bs z_ns c h w",
            bs=imgs.shape[0],
            c=imgs.shape[1],
            h=imgs.shape[2],
            w=imgs.shape[3],
        )

    imgs = [wandb.Image(x) for x in imgs[:num_samples_to_log]]

    if reconstructions.shape[1] != 1 and num_samples_per_z_to_consider != 1:
        reconstructions = [
            [wandb.Image(x_hat) for x_hat in rec_set[:num_samples_per_z_to_consider]]
            for rec_set in reconstructions[:num_samples_to_log]
        ]
    else:
        reconstructions = [wandb.Image(x_hat) for x_hat in reconstructions[:num_samples_to_log]]

    columns = ["step", "ids", "label", "x", "x_hat"]
    data = [
        [step, _id, label, x, x_hat]
        for _id, label, x, x_hat in zip(
            ids[:num_samples_to_log],
            labels[:num_samples_to_log],
            imgs,
            reconstructions,
        )
    ]
    return columns, data


def get_text_rec_table_data(
    ids, input_ids, input_text, reconstructions, step, num_samples_to_log, num_samples_per_z_to_consider
):
    if len(reconstructions) != input_ids.shape[0]:
        batch_size = input_ids.shape[0]
        num_samples_per_input = len(reconstructions) // batch_size
        reconstructions = [
            reconstructions[i : i + num_samples_per_input] for i in range(0, len(reconstructions), batch_size)
        ]

    input_ids = input_ids[:num_samples_to_log]
    input_text = input_text[:num_samples_to_log]

    if isinstance(reconstructions[0], list):
        reconstructions = [rec_set[:num_samples_per_z_to_consider] for rec_set in reconstructions[:num_samples_to_log]]
    else:
        reconstructions = reconstructions[:num_samples_to_log]

    columns = ["step", "id", "input_ids", "input_text", "output_text"]
    data = [
        [step, _id, tokenized_x, textual_x, output_text]
        for _id, tokenized_x, textual_x, output_text in zip(
            ids[:num_samples_to_log],
            input_ids,
            input_text,
            reconstructions,
        )
    ]
    return columns, data


def add_column_to_table_data(columns, data, new_col_name, new_col_data):
    # TODO: Verify correctness
    columns.append(new_col_name)
    for row, new_cell_value in zip(data, new_col_data):
        row.append(new_cell_value)
