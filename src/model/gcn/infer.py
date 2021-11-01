import torch
from pytorch_lightning import Trainer
from tqdm import tqdm
import re
from transformers import AutoTokenizer
import pandas as pd
import resource
from argparse import ArgumentParser


from src.model.gcn.qa_model_gcn import GraphQaModel
from src.model.gcn.data import GraphQaDataModule, rev_label_dict, InfluenceGraphNNData
from src.model.gcn.influence_graph import InfluenceGraph

def load_model(ckpt, graphs_file_name):

    model = GraphQaModel.load_from_checkpoint(ckpt).cuda()
    model.eval()
    trainer = Trainer(gpus=1)
    dm = GraphQaDataModule(basedir=model.hparams.dataset_basedir, tokenizer_name=model.hparams.model_name,
                           batch_size=model.hparams.batch_size,  graphs_file_name=graphs_file_name,)
    graphs = read_graphs(model.hparams.dataset_basedir, graphs_file_name)
    return model, trainer, dm, graphs


def eval(model, dataloader, influence_graphs, extract_paths: bool = False, paths_output_loc: str = None):

    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)
    true_paths, actual_path = [], []
    smax = torch.nn.Softmax(dim=-1)
    accs = []
    probs = []
    graph_expert_probs = []
    use_augmentation_probs = []
    predicted_labels, true_labels, Y_nodes = [], [], []
    total_evaluated, total_correct = 0., 0.
    with torch.no_grad():
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            tokens, token_type_ids, tokens_mask, graphs, labels = batch
            attention, logits, acc = model([data.cuda() if isinstance(data, torch.Tensor) else data for data in batch ])
            probs.extend(smax(logits)[:, 1].detach().cpu().tolist())
            accs.append(acc)
            batch_predicted_labels = torch.argmax(logits, -1)
            predicted_labels.extend(batch_predicted_labels.tolist())

            true_labels.extend(labels.tolist())
            total_evaluated += len(batch[0])

            total_correct += (acc.item() * len(batch[0]))

            print(
                f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}, Batch accuracy = {round(acc.item(), 2)}")
        print(
            f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}")
        # print(f"Accuracy = {round(np.array(accs).mean(), 2)}")

    data = pd.DataFrame({"idx": list(range(len(predicted_labels))),  "predicted_labels": predicted_labels,
                  "true_labels": true_labels,  "intense_probs": probs})

def read_graphs(basedir, graphs_file_name):
  print(basedir, graphs_file_name)
  influence_graphs = pd.read_json(
      f"{basedir}/{graphs_file_name}", orient='records', lines=True).to_dict(orient='records')
  graphs = {}
  for graph_dict in tqdm(influence_graphs, desc="Reading graphs", total=len(influence_graphs)):
      graphs[str(graph_dict["graph_id"])] = InfluenceGraph(graph_dict)
  return graphs


def verbose_inference(model, batch, influence_graphs, tokenizer, attention, attention_logits, logits, acc):
    if model.hparams.add_attention_supervision:
      question_tokens, question_type_ids, question_masks, graphs, labels, _ = batch
    else:
      question_tokens, question_type_ids, question_masks, graphs, labels = batch
    true_paths, actual_path = [], []
    for i in range(attention.shape[0]):
        tp, ap = infer_example_verbose(graphs[i], influence_graphs[graphs[i].graph_id],
                                       labels[i], question_tokens[i], question_type_ids[i], attention[i], logits[i], tokenizer)
        if tp:
            true_paths.append(tp)
            actual_path.append(ap)
    return true_paths, actual_path


def infer_example_verbose(graph, igraph, label, question, token_types, attention, logit, tokenizer, k=8):

    paragraph = tokenizer.decode(
        question * token_types, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    question = tokenizer.decode(question * (1 - token_types),
                                skip_special_tokens=True, clean_up_tokenization_spaces=True)
    cause, effect = extract_cause_effect_from_ques(question)
    nodes = decode_graph_nodes(graph, tokenizer)
    cause_node = find_node(cause, nodes)
    effect_node = find_node(effect, nodes)
    indices = attention[0].topk(k).indices.detach().cpu().numpy()
    most_attended_nodes = "\n".join(
        [f"{nodes[i]} ({round(attention[0][i].item() * 100, 2)})" for i in indices])
    predicted_answer = rev_label_dict[logit.argmax().item()]
    true_answer = rev_label_dict[label.item()]
    true_paths, actual_path = "", ""
    if cause_node and effect_node:
        true_paths = igraph.get_paths(cause_node)
        true_paths = " || ".join(list(set(
            ["->".join(path.nodes) for path in true_paths if path.nodes[-1] == effect_node])))
        actual_path = "->".join([InfluenceGraphNNData.index_node[i]
                                 for i in indices[:3]])

    verbose = True
    if verbose:
        print(igraph.to_ascii_drawing())
        print(f"# Paragraph: {paragraph}\n")
        print(f"# Question: {question}")
        print(f"# Question path: {cause_node} -> {effect_node} ({true_paths})")
        print(f"# True answer: {true_answer}")
        print(f"# Predicted answer: {predicted_answer} ({actual_path})\n")
        print(f"# Top {k} most attended nodes:\n{most_attended_nodes}")
        print("\n", "-" * 80, "\n")
    return true_paths, actual_path


def extract_cause_effect_from_ques(question: str):
    """Extracts cause and effect given questions
    'suppose the coal is carefully selected happens, how will it affect If less coal is broken down.'
    Arguments:
        question {[str]} -- [description]
    """
    question = question.lower()
    try:
      cause = re.search('suppose (.*)happens', question).group(1).strip()
    except:
      cause = ""
    try:
      effect = re.search('how will it affect (.*)',
                       question).group(1).strip()
    except:
      effect = ""
    return (cause, effect)


def decode_graph_nodes(graph, tokenizer):
    nodes = []

    for i, node_tokens in enumerate(graph.tokens):
        node_label = InfluenceGraphNNData.index_node[i]
        nodes.append(
            f"- {node_label}: {tokenizer.decode(node_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)}")
    return nodes


def find_node(x, nodes):
    for node in nodes:
        if x in node or x[:-1] in node:
            node = node.split(":")[0].strip()
            node = re.sub("-", "", node).strip()
            return node


if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    parser = ArgumentParser()
    parser.add_argument('--extract_paths', action='store_true')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--paths_output_loc', type=str, default="")
    parser.add_argument("--graphs_file_name", help="File that contains the generated influence graphs.", type=str)
    args = parser.parse_args()
    model, trainer, dm, graphs = load_model(args.ckpt, args.graphs_file_name)
    eval(model, dm.test_dataloader(), influence_graphs=graphs, extract_paths=args.extract_paths,
         paths_output_loc=args.paths_output_loc)