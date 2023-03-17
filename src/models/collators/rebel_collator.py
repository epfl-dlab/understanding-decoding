class RebelCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.params = kwargs

    def collate_fn(self, batch):
        collated_batch = {}

        for attr_name in "src", "tgt":
            if attr_name == "src":
                max_length = self.params["max_input_length"]

            elif attr_name == "tgt":
                max_length = self.params["max_output_length"]
            else:
                raise Exception(f"Unexpected attribute name `{attr_name}`!")

            tokenizer_output = self.tokenizer(
                [sample[attr_name] for sample in batch],
                return_tensors="pt",
                return_attention_mask=True,
                padding=self.params["padding"],
                max_length=max_length,
                truncation=self.params["truncation"],
            )

            for k, v in tokenizer_output.items():
                collated_batch["{}_{}".format(attr_name, k)] = v

        if self.params.get("target_padding_token_id", None) is not None:
            tgt_input_ids = collated_batch["tgt_input_ids"]
            tgt_input_ids.masked_fill_(
                tgt_input_ids == self.tokenizer.pad_token_id, self.params["target_padding_token_id"]
            )

        collated_batch["raw"] = batch

        return collated_batch
