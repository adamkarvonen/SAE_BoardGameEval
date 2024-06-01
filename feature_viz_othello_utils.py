import torch as t
import circuits.othello_utils as othello_utils
import matplotlib.pyplot as plt
import gc
from IPython.display import HTML, display
from tqdm import trange
from typing import Optional
from circuits.othello_engine_utils import OthelloBoardState, itos, to_board_label


def convert_othello_dataset_sample_to_board(sample_i, move_idx=None):
    if type(sample_i) == t.Tensor:
        sample_i = sample_i.tolist()
    context = [othello_utils.itos[s] for s in sample_i]
    if move_idx is not None:
        context = context[: move_idx + 1]
    board_state_RR = othello_utils.games_batch_to_state_stack_mine_yours_BLRRC([context])[0][-1]
    board_state_RR = t.argmax(board_state_RR, dim=-1) - 1
    return board_state_RR


def plot_othello_board_highlighted(ax, true_board_RR, bg_board_RR=None, title=""):
    """
    Plots a comparison of the true and reconstructed Othello boards using matplotlib.

    Args:
    true_board (torch.Tensor): A 2D tensor representing the true Othello board.
    recon_board (torch.Tensor): A 2D tensor representing the reconstructed Othello board.
    """

    true_color_map = {-1: "black", 0: "white", 1: "cornsilk"}
    if bg_board_RR is None:
        bg_board_RR = t.zeros_like(true_board_RR)
        cmap = plt.matplotlib.colors.ListedColormap(["white"])
        print_color_bar = False
        vmin = 0
        vmax = 0
    else:
        bg_max_abs = t.abs(bg_board_RR).max().item()
        if bg_board_RR.min() < 0:
            cmap = "RdBu"
            vmin = -bg_max_abs
            vmax = bg_max_abs
        else:
            cmap = "Blues"
            vmin = 0
            vmax = bg_board_RR.max().item()
        print_color_bar = True
        if isinstance(bg_board_RR, t.Tensor):
            bg_board_RR = bg_board_RR.cpu().detach().numpy()

    ax.imshow(bg_board_RR, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(8):
        for j in range(8):
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=0.5)
            )
            if true_board_RR[i, j].item() == 0:
                continue
            circle = plt.Circle(
                (j, i), 0.3, color=true_color_map[true_board_RR[i, j].item()], fill=True
            )
            circle_edges = plt.Circle((j, i), 0.3, color="black", fill=False)
            ax.add_artist(circle)
            ax.add_artist(circle_edges)
    ax.set_xticks(range(8))
    ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H"])
    ax.set_title(title)
    if print_color_bar:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)


# Test visualization
# random_bg = t.arange(-10,54).reshape(8,8)
# plot_othello_board_highlighted(sample_board, random_bg)


# functions for visualizing highlighted token sequences
def shade(value, max_value):
    if abs(value) > max_value:
        raise ValueError("Absolute value must be less than or equal to max_value.")

    epsilon = 1e-8
    normalized_value = value / (max_value + epsilon)

    if normalized_value < 0:
        # Red shade for negative values
        red = 255
        green = int(255 * (1 + normalized_value))
        blue = int(255 * (1 + normalized_value))
    else:
        # Blue shade for positive values
        red = int(255 * (1 - normalized_value))
        green = int(255 * (1 - normalized_value))
        blue = 255

    # White color for zero value
    if value == 0:
        red = green = blue = 255

    # Convert RGB values to hex color code
    hex_color = "#{:02x}{:02x}{:02x}".format(red, green, blue)

    return hex_color


def visualize_game_seq(context_i, activations, max_value, prefix=""):
    context_s = [othello_utils.itos[s] for s in context_i]
    labeled_seq = list(map(othello_utils.to_board_label, context_s))
    html_elements = []
    for token, act in zip(labeled_seq, activations):
        hex_color = shade(act, max_value)
        s = token
        s = s.replace(" ", "&nbsp;")
        html_element = f'<span style="background-color: {hex_color}; color: black">{s}</span>'
        html_elements.append(html_element)

    combined_html = " ".join(html_elements)
    combined_html = prefix + combined_html
    return HTML(combined_html)


def cossim_logit_feature_decoder(
    model, ae, feat_idx: int, node_type: str, layer: Optional[int] = None
):
    if node_type == "sae_feature":
        d_model_vec = ae.decoder.weight[:, feat_idx]
    elif node_type == "mlp_neuron":
        if layer == None:
            raise ValueError("Must specify layer for MLP neuron")
        d_model_vec = model.blocks[layer].mlp.W_out[feat_idx, :]
    else:
        raise ValueError(f"Unknown node type {node_type}")
    cossim = t.cosine_similarity(
        d_model_vec, model.W_U[:, 1:].T, dim=1
    )  # NOTE 0 is a special token?
    return cossim


def cossim_tokenembed_feature_decoder(
    model, ae, feat_idx: int, node_type: str, layer: Optional[int] = None
):
    if node_type == "sae_feature":
        d_model_vec = ae.decoder.weight[:, feat_idx]
    elif node_type == "mlp_neuron":
        if layer == None:
            raise ValueError("Must specify layer for MLP neuron")
        d_model_vec = model.blocks[layer].mlp.W_out[feat_idx, :]
    else:
        raise ValueError(f"Unknown node type {node_type}")
    cossim = t.cosine_similarity(d_model_vec, model.W_E[1:, :], dim=1)
    return cossim


def convert_vocab_to_board_tensor(vocab_vals, device) -> t.Tensor:
    ll_board = t.zeros(64, device=device)
    ll_board[othello_utils.stoi_indices] = vocab_vals
    ll_board = ll_board.view(8, 8)
    return ll_board


def convert_seq_to_board_tensor(seq, game, device) -> t.Tensor:
    """The game sequence is a list of token indices, where each token is a single integer. We have to sort by the board indices to get the correct board tensor."""
    # map from token index to board index
    ll_board = t.zeros(64, device=device)
    game_s = [othello_utils.itos[i] for i in game]
    ll_board[game_s] = seq
    ll_board = ll_board.view(8, 8)
    return ll_board


def visualize_vocab(ax, vocab_vals, device, title=""):
    ll_board = convert_vocab_to_board_tensor(vocab_vals, device)
    cmap = "RdBu"
    vmin = -vocab_vals.abs().max().item()
    vmax = vocab_vals.abs().max().item()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(ll_board.cpu().detach().numpy(), cmap=cmap, norm=norm)

    # Specify the number of ticks for the colorbar using linspace
    ticks = t.linspace(vmin, vmax, 5)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, ticks=ticks, fraction=0.046, pad=0.04
    )
    cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])
    ax.set_xticks(range(8))
    ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H"])
    ax.set_title(title)

def visualize_board_from_tensor(ax, board_tensor, title="",):
    ll_board = board_tensor.view(8, 8)
    cmap = "Blues"
    vmin = 0
    vmax = board_tensor.abs().max().item()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(ll_board.cpu().detach().numpy(), cmap=cmap, norm=norm)

    # Specify the number of ticks for the colorbar using linspace
    # ticks = t.linspace(vmin, vmax, 4)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, fraction=0.046, pad=0.04
    )
    # cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])
    ax.set_xticks(range(8))
    ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H"])
    ax.set_title(title)

def visualize_lens(
    ax,
    model,
    ae,
    feat_idx: int,
    node_type: str,
    layer: Optional[int],
    cossim_func,
    device,
    title: str = "",
):
    cossim = cossim_func(model, ae, feat_idx, node_type, layer)
    visualize_vocab(ax, cossim, title=title, device=device)


def plot_lenses(model, ae, feat_idx: int, device: str, node_type: str, layer: Optional[int] = None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    visualize_lens(
        axs[0],
        model,
        ae,
        feat_idx,
        node_type,
        layer,
        cossim_tokenembed_feature_decoder,
        title="with token_embed",
        device=device,
    )
    visualize_lens(
        axs[1],
        model,
        ae,
        feat_idx,
        node_type,
        layer,
        cossim_logit_feature_decoder,
        title="with unembed (DLA)",
        device=device,
    )
    fig.suptitle(f"Cosine sim of #feature {feat_idx} decoder")
    plt.show()


# Currently not used bc batch computing for all features doesn't fit in memory
# for given sae and games_batch, get the activations and attributions

# def get_acts_attrs_VBSF(model, ae, games, submodule, device, compute_ie_embed=False):
#     # Get feature acts and logit grads per feature
#     logit_grads_per_feature_VNSF = t.zeros(
#         (model.cfg.d_vocab_out, *games.shape, hidden_dim),
#         device=device,
#         dtype=t.bfloat16
#     )
#     if compute_ie_embed:
#         feature_grads_per_tokenembed_FBSV = t.zeros(
#             (hidden_dim, *games.shape, model.cfg.d_model),
#             device=device,
#             dtype=t.bfloat16
#         )
#     with model.trace(games):
#         embed_acts_NSF = model.hook_embed.output.save()
#         x = submodule.output
#         f = ae.encode(x)
#         feature_acts_NSF = f.save()
#         submodule.output = ae.decode(f)

#         if compute_ie_embed:
#             # Cache grads of features wrt each tokenembed
#             for feat_idx in range(hidden_dim):
#                 model.zero_grad()
#                 feature_grads_per_tokenembed_FBSV[feat_idx] = model.hook_embed.output.grad.save()
#                 f[...,feat_idx].sum().backward(retain_graph=True)

#         # Cache grads of logits wrt each feature
#         for vocab_idx in range(model.cfg.d_vocab_out):
#             model.zero_grad()
#             logit_grads_per_feature_VNSF[vocab_idx] = f.grad.save()
#             model.unembed.output[..., vocab_idx].sum().backward(retain_graph=True)

#     if compute_ie_embed:
#         # Approximate indirect effect of tokenembeds on features via attribution patching (mean ablation)
#         mean_embed = embed_acts_NSF.value.mean(dim=(0,1))
#         ie_tokenembed_to_features_FBSV = (feature_grads_per_tokenembed_FBSV * (embeds - mean_embed))
#         ie_tokenembed_to_features_VBF = ie_tokenembed_to_features_FBSV.sum(dim=-2).permute(2, 1, 0)


#     # Approximate indirect effect of features on logits via attribution patching (zero ablation)
#     ie_feature_to_logits_VBSF = logit_grads_per_feature_VNSF * feature_acts_NSF
#     # Each token occurs exactly once in the sequence, so we can sum over the sequence dim
#     ie_feature_to_logits_VBF = ie_feature_to_logits_VBSF.sum(dim=-2)

#     # Sort acts per tokenembed
#     acts_per_tokenembed_VNF = t.zeros(
#     (model.cfg.d_vocab, games.shape[0], hidden_dim),
#     device=device
#     )
#     for game_idx, game in enumerate(games):
#         acts_per_tokenembed_VNF[game, game_idx] = feature_acts_NSF[game_idx]

#     del feature_acts_NSF
#     del embed_acts_NSF
#     del logit_grads_per_feature_VNSF
#     del ie_feature_to_logits_VBSF
#     if compute_ie_embed:
#         del feature_grads_per_tokenembed_FBSV
#         del ie_tokenembed_to_features_FBSV
#     t.cuda.empty_cache()
#     gc.collect()

#     if compute_ie_embed:
#         return acts_per_tokenembed_VNF, ie_feature_to_logits_VBF, ie_tokenembed_to_features_VBF
#     else:
#         return acts_per_tokenembed_VNF, ie_feature_to_logits_VBF

# Compute once
# games_batch = buffer.token_batch(batch_size=games_batch_size)
# # acts_per_tokenembed_VNF, ie_feature_to_logits_VBF, ie_tokenembed_to_features_VBF = get_acts_attrs_VBSF(model, ae, games_batch, submodule compute_ie_embed=True)
# acts_per_tokenembed_VNF, ie_feature_to_logits_VBF = get_acts_attrs_VBSF(model, ae, games_batch, submodule, compute_ie_embed=False)

# print(f'games_batch shape: {games_batch.shape}')
# print(f'acts shape: {acts_per_tokenembed_VNF.shape}')
# print(f'ie_feature_to_logits shape: {ie_feature_to_logits_VBF.shape}')
# # print(f'ie_tokenembed_to_features shape: {ie_tokenembed_to_features_VBF.shape}')


# for given sae and games_batch, get the activations and attributions
def get_acts_IEs_VN(model, ae, game_batches_LBS, submodule, feat_idx, device, compute_ie=False):
    # Get feature acts and logit grads per feature
    # L = game_batches_LBS.shape[0]/ n batches, B = game_batches_LBS.shape[1]/ games per batch
    # V = D_vocab(_out), N = L*B, S = n_ctx, F = D_feat
    # True shape during initialization is L,B instead of N = L*B

    # this works for both SAE features and MLP neurons, referred to as "nodes"
    if ae is not None:
        # get results for SAE features
        hidden_dim = ae.dict_size
    else:
        # get results for MLP neurons
        hidden_dim = model.cfg.d_mlp

    node_acts_NSF = t.zeros((*game_batches_LBS.shape, hidden_dim), device=device)
    embed_acts_NSF = t.zeros((*game_batches_LBS.shape, model.cfg.d_model), device=device)
    if compute_ie:
        logit_grads_per_node_VNS = t.zeros(
            (model.cfg.d_vocab_out, *game_batches_LBS.shape), device=device
        )
        node_grads_per_tokenembed_NSM = t.zeros(
            (*game_batches_LBS.shape, model.cfg.d_model), device=device
        )

    L = game_batches_LBS.shape[0]
    N = L * game_batches_LBS.shape[1]
    for batch_idx in trange(L, desc="Games batch"):
        with model.trace(game_batches_LBS[batch_idx]):
            embed_acts_NSF[batch_idx] = model.hook_embed.output.save()

            if ae is not None:
                x = submodule.output
                f = ae.encode(x)
                submodule.output = ae.decode(f)
                node_acts_NSF[batch_idx] = f.save()
            else:
                f = submodule.output
                node_acts_NSF[batch_idx] = f.save()

            if compute_ie:
                f.retain_grad()
                # Cache grads of nodes wrt each tokenembed
                model.zero_grad()
                node_grads_per_tokenembed_NSM[batch_idx] = model.hook_embed.output.grad.save()
                f[..., feat_idx].sum().backward(retain_graph=True)

                # Cache grads of logits wrt each node
                for vocab_idx in range(model.cfg.d_vocab_out):
                    model.zero_grad()
                    logit_grads = f.grad.save()
                    logit_grads_per_node_VNS[vocab_idx][batch_idx] = logit_grads[..., feat_idx]
                    model.unembed.output[..., vocab_idx].sum().backward(retain_graph=True)

    # Reshape to collapse the batch dim
    node_acts_NSF = node_acts_NSF.view(N, model.cfg.n_ctx, hidden_dim)
    node_acts_NS = node_acts_NSF[..., feat_idx]
    del node_acts_NSF

    # Sort per token (instead of sequence pos)
    acts_per_tokenembed_VN = t.zeros((model.cfg.d_vocab, N), device=device)
    for game_idx, game in enumerate(game_batches_LBS.view(N, -1)):
        acts_per_tokenembed_VN[game, game_idx] = node_acts_NS[game_idx]

    if compute_ie:
        # Reshape to collapse the batch dim
        embed_acts_NSF = embed_acts_NSF.view(N, model.cfg.n_ctx, model.cfg.d_model)
        logit_grads_per_node_VNS = logit_grads_per_node_VNS.view(
            model.cfg.d_vocab_out, N, model.cfg.n_ctx
        )
        node_grads_per_tokenembed_NSM = node_grads_per_tokenembed_NSM.view(
            N, model.cfg.n_ctx, model.cfg.d_model
        )

        # Approximate indirect effect of nodes on logits via attribution patching (zero ablation), grad * (patch - clean)
        # Each token occurs exactly once in the sequence, so we can sum over the sequence dim
        ie_node_to_logits_VN = (logit_grads_per_node_VNS * (0 - node_acts_NS)).sum(dim=-1)
        del logit_grads_per_node_VNS
        # Approximate indirect effect of tokenembeds on nodes via attribution patching (mean ablation), grad * (patch - clean)
        mean_embed_F = embed_acts_NSF.mean(dim=(0, 1))
        ie_tokenembed_to_nodes_NS = (
            node_grads_per_tokenembed_NSM * (mean_embed_F - embed_acts_NSF)
        ).sum(dim=-1)
        del embed_acts_NSF
        del mean_embed_F

        # Sort per token (instead of sequence pos)
        ie_tokenembed_to_nodes_VN = t.zeros((model.cfg.d_vocab, N), device=device)
        for game_idx, game in enumerate(game_batches_LBS.view(N, -1)):
            ie_tokenembed_to_nodes_VN[game, game_idx] = ie_tokenembed_to_nodes_NS[game_idx]
        del ie_tokenembed_to_nodes_NS
        del node_grads_per_tokenembed_NSM

    t.cuda.empty_cache()
    gc.collect()

    if compute_ie:
        return node_acts_NS, acts_per_tokenembed_VN, ie_node_to_logits_VN, ie_tokenembed_to_nodes_VN
    return node_acts_NS, acts_per_tokenembed_VN, None, None


def plot_mean_metrics(
    acts_per_tokenembed_VN,
    ie_tokenembed_to_features_VN,
    ie_feature_to_logits_VN,
    feat_idx,
    n_games,
    with_ie,
    device,
):
    if with_ie:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        mean_ie_tokenembed_to_features_V = ie_tokenembed_to_features_VN.mean(dim=-1)
        mean_logit_attributions_V = ie_feature_to_logits_VN.mean(dim=-1)
        visualize_vocab(
            axs[1],
            mean_ie_tokenembed_to_features_V[1:],
            title=f"IE token_embed --> feature\n(AP, mean ablation)",
            device=device,
        )
        visualize_vocab(
            axs[2],
            mean_logit_attributions_V[1:],
            title=f"IE feature --> logits\n(AP, zero ablation)",
            device=device,
        )
        ax0 = axs[0]
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))

    mean_feature_activations_tokenembed_V = acts_per_tokenembed_VN.mean(dim=-1)
    visualize_vocab(
        ax0, mean_feature_activations_tokenembed_V[1:], title=f"Feature activations", device=device
    )
    fig.suptitle(f"Mean metrics over {n_games} games for feature #{feat_idx}")
    plt.show()


def plot_max_act_boardstate(
    ax, game_idx, games_batch_NS, feature_acts_NS, max_act_pos, title, device
):
    game = games_batch_NS[game_idx][: max_act_pos + 1].tolist()
    acts = feature_acts_NS[game_idx][: max_act_pos + 1]
    true_board_RR = convert_othello_dataset_sample_to_board(game)
    act_board_RR = convert_seq_to_board_tensor(acts, game, device)
    plot_othello_board_highlighted(ax, true_board_RR, bg_board_RR=act_board_RR, title=title)
    plt.show()


def plot_top_k_games(
    feat_idx,
    games_batch_NS,
    feature_acts_NS,
    acts_per_tokenembed_VN,
    ie_tokenembed_to_features_VN,
    ie_feature_to_logits_VN,
    device,
    sort_metric="activation",
    k=10,
    with_ie=False,
):
    if (with_ie is False) or (sort_metric == "activation"):
        top_game_indices = acts_per_tokenembed_VN.abs().max(dim=0)[0].topk(k).indices
    elif sort_metric == "ie_embed":
        top_game_indices = ie_tokenembed_to_features_VN.abs().max(dim=0)[0].topk(k).indices
    elif sort_metric == "ie_logit":
        top_game_indices = ie_feature_to_logits_VN.abs().max(dim=0)[0].topk(k).indices

    for i, game_idx in enumerate(top_game_indices):
        print(f"Top {i+1} {sort_metric} game:")

        game = games_batch_NS[game_idx].tolist()
        feature_act_S = feature_acts_NS[game_idx]
        max_abs_act, max_act_pos = feature_act_S.abs().max(dim=0)
        max_act_square = othello_utils.to_board_label(othello_utils.itos[game[max_act_pos]])

        display(
            visualize_game_seq(game, feature_act_S, max_abs_act, prefix="feature activations: <br>")
        )

        if with_ie:
            ie_embed_V = ie_tokenembed_to_features_VN[:, game_idx]
            ie_logit_V = ie_feature_to_logits_VN[:, game_idx]
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            visualize_vocab(
                axs[1],
                ie_embed_V[1:],
                title=f"IE token_embed --> feature\n(AP, mean ablation)",
                device=device,
            )
            visualize_vocab(
                axs[2],
                ie_logit_V[1:],
                title=f"IE feature --> logits\n(AP, zero ablation)",
                device=device,
            )
            ax0 = axs[0]
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=(5, 5))

        plot_max_act_boardstate(
            ax0,
            game_idx,
            games_batch_NS,
            feature_acts_NS,
            max_act_pos,
            title=f"Feature activations on input tokens\nat move #{max_act_pos}: {max_act_square}",
            device=device,
        )
        fig.suptitle(f"Top {i+1} {sort_metric} game for feature #{feat_idx}")
        plt.show()


# Play through a game
class BoardPlayer():
    def __init__(self, game):
        self.game_i = game.cpu().numpy()
        self.game_s = [itos[move] for move in self.game_i]
        self.board = OthelloBoardState()
        self.cur_idx = 0

    def _display(self):
        cur_move_highlighted = t.zeros(64)
        cur_move_highlighted[self.game_s[self.cur_idx]] = 1
        cur_move_highlighted = cur_move_highlighted.view(8,8)
        highlight_seq = t.zeros(59)
        highlight_seq[self.cur_idx] = 1

        display(visualize_game_seq(self.game_i, highlight_seq, 2, prefix="Current move: <br>"))
        fig, ax = plt.subplots()
        move_label = to_board_label(self.game_s[self.cur_idx])
        plot_othello_board_highlighted(ax, self.board.state, bg_board_RR=cur_move_highlighted, title=f'Move {self.cur_idx}: {move_label}')
        plt.show()

    def next(self):
        if self.cur_idx < 59:
            move = self.game_s[self.cur_idx]
            self.board.umpire(move)
            self._display()
            self.cur_idx += 1
        else:
            print('Last move reached.')

    def prev(self):
        if self.cur_idx > 0:
            self.board = OthelloBoardState()
            for move in self.game_s[:self.cur_idx]:
                self.board.umpire(move)
            self._display()
            self.cur_idx -= 1
        else:
            print('First move reached.')