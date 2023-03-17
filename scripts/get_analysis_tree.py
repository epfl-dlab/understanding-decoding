import hydra
import sys

import torch
from src.utils.hydra_ad_hoc import get_config
from src.mdp.trees import AnalysisTree, LanguageModelAsTree
from src.mdp.utils import draw_graph, save_graph

get_config_params = {"config_name": "train_root", "work_dir": "../",
                     "data_dir": "data/"}
overrides = [
    "+evaluation=genie_cie_large",
    "model/decoding=[genie_generic, beam_search]",
    f"+evaluation_model=cie_oracle",
    # "debug=fast"
]
config = get_config(**get_config_params, overrides=overrides)
configs_folder = "../configs"

with hydra.initialize(version_base="1.2", config_path=configs_folder):
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)
    model = hydra.utils.instantiate(config.model, datamodule=datamodule)
    # Initialize the evaluation model (i.e. the value function used during decoding)
    lm_as_tree = LanguageModelAsTree(model, model.tokenizer)
    evaluation_model = hydra.utils.instantiate(config.evaluation_model, tree=lm_as_tree)
    model.evaluation_model = evaluation_model

model.hparams.decoding.hf_generation_params.num_beams = 6
model.hparams.decoding.hf_generation_params.num_return_sequences = 6

datamodule.setup(stage="test")
test_dl = datamodule.test_dataloader()
for batch in test_dl:
    break

input_text = batch["src"][:1]
raw_output = batch["tgt"][:1]

model_kwargs = model.tokenize(model.model, model.tokenizer, input_text, raw_output)
src_inputs = [
    {"input_ids": model_kwargs['input_ids'], "attention_mask": model_kwargs['attention_mask']}
]
tgt_inputs = [
    {"input_ids": model_kwargs['decoder_input_ids']}
]

tgt_scores_output = model.compute_scores(src_inputs, tgt_inputs, None, True)
tgt_per_token_scores = tgt_scores_output['scores_obj'][0].get_processed_score(return_final_score_only=False)

results = model.sample(input_text,
                       return_generation_outputs=True,
                       output_scores=True)

beam_search_beams = results['generation_outputs'].sequences
prediction_src = src_inputs * model.hparams.decoding.hf_generation_params.num_return_sequences
prediction_tgt = [
    {"input_ids": seq.clone()}
    for seq
    in beam_search_beams
]
prefix_allowed_tokens_fn = model.get_prefix_allowed_fn(input_text, None, None)

pred_scores_output = model.compute_scores(prediction_src, prediction_tgt, None, False)
pred_per_token_scores = [scores_obj.get_processed_score(return_final_score_only=False) for scores_obj in
                         pred_scores_output['scores_obj']]


def get_per_token_values(ids, target_ids):
    if ids.dim() == 1:
        ids = ids[None, :]

    assert target_ids.dim() == 2
    assert ids.dim() == 2

    values = []
    for i in range(2, ids.shape[-1] + 1):
        val = get_value(input_ids=ids[:, :i], target_ids=target_ids)
        values.append(val)

    return torch.tensor(values)


def get_value(input_ids, target_ids):
    return evaluation_model.evaluate(input_ids=input_ids[:, :-1],
                                     target_ids=target_ids,
                                     next_token_ids=input_ids[:, -1][:, None]).item()


def input_ids_to_node_id(input_ids, pad_token=None):
    if pad_token is None:
        return tuple(input_ids.squeeze().numpy())

    attention_mask = input_ids.squeeze() != pad_token
    return tuple(input_ids.squeeze()[attention_mask].numpy())



target_ids = tgt_inputs[0]['input_ids']
tgt_values = get_per_token_values(target_ids, target_ids)

pred_values = [get_per_token_values(pred['input_ids'], target_ids) for pred in prediction_tgt]
print(tgt_values)
print(pred_values[0])
print(pred_values[1])
print(pred_values[2])

# plot
# Make sure that TGT has the EOS token
a_tree = AnalysisTree(lm_as_tree)

a_tree.add_path(input_ids_to_node_id(tgt_inputs[0]['input_ids']),
                tgt_per_token_scores,
                tgt_values,
                label="TGT", terminal=True)

c = 1
for pred, scores, values in zip(prediction_tgt, pred_per_token_scores, pred_values):
    a_tree.add_path(node_id=input_ids_to_node_id(pred['input_ids'], model.model.config.pad_token_id),
                    scores=scores,
                    values=values[:len(scores)],
                    label=f"P-{c}",
                    terminal=True)
    c += 1

viz_params = {'decode_node_id': False,
              'decode_action_id': False,
              'mark_terminal_nodes': True}

graph = a_tree.visualize_tree(**viz_params)

draw_graph(graph)
save_graph(graph, "genie_oracle_bs.png")
