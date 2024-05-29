import torch as t
import circuits.othello_utils as othello_utils
import matplotlib.pyplot as plt
import gc
from IPython.display import HTML, display



def convert_othello_dataset_sample_to_board(sample_i, move_idx=None):
    if type(sample_i) == t.Tensor:
        sample_i = sample_i.tolist()
    context = [othello_utils.itos[s] for s in sample_i]
    if move_idx is not None:
        context = context[:move_idx+1]
    board_state_RR = othello_utils.games_batch_to_state_stack_mine_yours_BLRRC([context])[0][-1]
    board_state_RR = t.argmax(board_state_RR, dim=-1) - 1
    return board_state_RR

def plot_othello_board_highlighted(ax, true_board_RR, bg_board_RR=None, title=''):
    """
    Plots a comparison of the true and reconstructed Othello boards using matplotlib.

    Args:
    true_board (torch.Tensor): A 2D tensor representing the true Othello board.
    recon_board (torch.Tensor): A 2D tensor representing the reconstructed Othello board.
    """

    true_color_map = {-1: 'black', 0: 'white', 1: 'cornsilk'}
    if bg_board_RR is None:
        bg_board_RR = t.zeros_like(true_board_RR)
        cmap = plt.matplotlib.colors.ListedColormap(['white'])
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

    ax.imshow(bg_board_RR, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(8):
        for j in range(8):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=0.5))
            if true_board_RR[i, j].item() == 0:
                continue
            circle = plt.Circle((j, i), 0.3, color=true_color_map[true_board_RR[i, j].item()], fill=True)
            circle_edges = plt.Circle((j, i), 0.3, color='black', fill=False)
            ax.add_artist(circle)
            ax.add_artist(circle_edges)
    ax.set_xticks(range(8))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
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
    
    normalized_value = value / max_value
    
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

def visualize_game_seq(context_i, activations, max_value, prefix=''):
    context_s = [othello_utils.itos[s] for s in context_i]
    labeled_seq = list(map(othello_utils.to_board_label, context_s))
    html_elements = []
    for token, act in zip(labeled_seq, activations):
        hex_color = shade(act, max_value)
        s = token
        s = s.replace(' ', '&nbsp;')
        html_element = f'<span style="background-color: {hex_color}; color: black">{s}</span>'
        html_elements.append(html_element)
    
    combined_html = ' '.join(html_elements)
    combined_html = prefix + combined_html
    return HTML(combined_html)

def cossim_logit_feature_decoder(model, ae, feat_idx):
    feat_decoder_vec = ae.decoder.weight[:, feat_idx]
    cossim = t.cosine_similarity(feat_decoder_vec, model.W_U[:, 1:].T, dim=1) # NOTE 0 is a special token?
    return cossim

def cossim_tokenembed_feature_decoder(model, ae, feat_idx):
    feat_decoder_vec = ae.decoder.weight[:, feat_idx]
    cossim = t.cosine_similarity(feat_decoder_vec, model.W_E[1:, :], dim=1)
    return cossim

def visualize_vocab(ax, vocab_vals, device, title=''):
    ll_board = t.zeros(64, device=device)
    ll_board[othello_utils.stoi_indices] = vocab_vals
    ll_board = ll_board.view(8, 8)
    cmap = "RdBu"
    vmin = -vocab_vals.abs().max().item()
    vmax = vocab_vals.abs().max().item()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(ll_board.cpu().detach().numpy(), cmap=cmap, norm=norm)

    # Specify the number of ticks for the colorbar using linspace
    ticks = t.linspace(vmin, vmax, 5)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])
    ax.set_xticks(range(8))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    ax.set_title(title)

def visualize_lens(ax, model, ae, feat_idx, cossim_func, device, title=''):
    cossim = cossim_func(model, ae, feat_idx)
    visualize_vocab(ax, cossim, title=title, device=device)

def plot_lenses(model, ae, feat_idx, device):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        
    visualize_lens(axs[0], model, ae, feat_idx, cossim_tokenembed_feature_decoder, title='with token_embed', device=device)
    visualize_lens(axs[1], model, ae, feat_idx, cossim_logit_feature_decoder, title='with unembed (logit lens)', device=device)
    fig.suptitle(f'Cosine sim of #feature {feat_idx} decoder')
    plt.show()


# for given sae and games_batch, get the activations and attributions
# Currently not used bc batch computing for all features doesn't fit in memory
def get_acts_attrs_VBSF(model, ae, games, submodule, device, compute_ie_embed=False):
    # Get feature acts and logit grads per feature
    logit_grads_per_feature_VBSF = t.zeros(
        (model.cfg.d_vocab_out, *games.shape, ae.dict_size),
        device=device,
        dtype=t.bfloat16
    )
    if compute_ie_embed:
        feature_grads_per_tokenembed_FBSV = t.zeros(
            (ae.dict_size, *games.shape, model.cfg.d_model),
            device=device,
            dtype=t.bfloat16
        )
    with model.trace(games): 
        embeds = model.hook_embed.output.save()
        x = submodule.output
        f = ae.encode(x)
        feature_acts_BSF = f.save() 
        submodule.output = ae.decode(f)

        if compute_ie_embed:
            # Cache grads of features wrt each tokenembed
            for feat_idx in range(ae.dict_size):
                model.zero_grad()
                feature_grads_per_tokenembed_FBSV[feat_idx] = model.hook_embed.output.grad.save()
                f[...,feat_idx].sum().backward(retain_graph=True)

        # Cache grads of logits wrt each feature
        for vocab_idx in range(model.cfg.d_vocab_out):
            model.zero_grad()
            logit_grads_per_feature_VBSF[vocab_idx] = f.grad.save()
            model.unembed.output[..., vocab_idx].sum().backward(retain_graph=True)

    if compute_ie_embed:
        # Approximate indirect effect of tokenembeds on features via attribution patching (mean ablation)
        mean_embed = embeds.value.mean(dim=(0,1))
        ie_tokenembed_to_features_FBSV = (feature_grads_per_tokenembed_FBSV * (embeds - mean_embed))
        ie_tokenembed_to_features_VBF = ie_tokenembed_to_features_FBSV.sum(dim=-2).permute(2, 1, 0)


    # Approximate indirect effect of features on logits via attribution patching (zero ablation)
    ie_feature_to_logits_VBSF = logit_grads_per_feature_VBSF * feature_acts_BSF
    # Each token occurs exactly once in the sequence, so we can sum over the sequence dim
    ie_feature_to_logits_VBF = ie_feature_to_logits_VBSF.sum(dim=-2)

    # Sort acts per tokenembed
    acts_per_tokenembed_VBF = t.zeros(
    (model.cfg.d_vocab, games.shape[0], ae.dict_size),
    device=device
    )
    for game_idx, game in enumerate(games):
        acts_per_tokenembed_VBF[game, game_idx] = feature_acts_BSF[game_idx]
    
    del feature_acts_BSF
    del embeds
    del logit_grads_per_feature_VBSF
    del ie_feature_to_logits_VBSF
    if compute_ie_embed:
        del feature_grads_per_tokenembed_FBSV
        del ie_tokenembed_to_features_FBSV
    t.cuda.empty_cache()
    gc.collect()

    if compute_ie_embed:
        return acts_per_tokenembed_VBF, ie_feature_to_logits_VBF, ie_tokenembed_to_features_VBF
    else:
        return acts_per_tokenembed_VBF, ie_feature_to_logits_VBF

# Compute once
# games_batch = buffer.token_batch(batch_size=games_batch_size)
# # acts_per_tokenembed_VBF, ie_feature_to_logits_VBF, ie_tokenembed_to_features_VBF = get_acts_attrs_VBSF(model, ae, games_batch, submodule compute_ie_embed=True)
# acts_per_tokenembed_VBF, ie_feature_to_logits_VBF = get_acts_attrs_VBSF(model, ae, games_batch, submodule, compute_ie_embed=False)

# print(f'games_batch shape: {games_batch.shape}')
# print(f'acts shape: {acts_per_tokenembed_VBF.shape}')
# print(f'ie_feature_to_logits shape: {ie_feature_to_logits_VBF.shape}')
# # print(f'ie_tokenembed_to_features shape: {ie_tokenembed_to_features_VBF.shape}')


# for given sae and games_batch, get the activations and attributions
def get_acts_IEs_VB(model, ae, game_batches_NBS, submodule, feat_idx, device, compute_ie_embed=False):
    # Get feature acts and logit grads per feature
    # True shape during initialization is [V NBS F]. We reshape after model.trace
    feature_acts_BSF = t.zeros((*game_batches_NBS.shape, ae.dict_size), device=device)
    embeds = t.zeros((*game_batches_NBS.shape, model.cfg.d_model), device=device)
    logit_grads_per_feature_VBS = t.zeros((model.cfg.d_vocab_out, *game_batches_NBS.shape), device=device)
    if compute_ie_embed:
        feature_grads_per_tokenembed_BSM = t.zeros((*game_batches_NBS.shape, model.cfg.d_model), device=device)
    
    n_batches = game_batches_NBS.shape[0]
    n_games_total = n_batches * game_batches_NBS.shape[1]
    for batch_idx in range(n_batches):
        with model.trace(game_batches_NBS[batch_idx]): 
            embeds[batch_idx] = model.hook_embed.output.save()
            x = submodule.output
            f = ae.encode(x)
            submodule.output = ae.decode(f)
            feature_acts_BSF[batch_idx] = f.save() 
            f.retain_grad()

            if compute_ie_embed:
                # Cache grads of features wrt each tokenembed
                model.zero_grad()
                feature_grads_per_tokenembed_BSM[batch_idx] = model.hook_embed.output.grad.save()
                f[..., feat_idx].sum().backward(retain_graph=True)

            # Cache grads of logits wrt each feature
            for vocab_idx in range(model.cfg.d_vocab_out):
                model.zero_grad()
                logit_grads = f.grad.save()
                logit_grads_per_feature_VBS[vocab_idx][batch_idx] = logit_grads[..., feat_idx]
                model.unembed.output[..., vocab_idx].sum().backward(retain_graph=True)

    # Reshape to collapse the batch dim
    feature_acts_BSF = feature_acts_BSF.view(n_games_total, model.cfg.n_ctx, ae.dict_size)
    embeds = embeds.view(n_games_total, model.cfg.n_ctx, model.cfg.d_model)
    logit_grads_per_feature_VBS = logit_grads_per_feature_VBS.view(model.cfg.d_vocab_out, n_games_total, model.cfg.n_ctx)
    if compute_ie_embed:
        feature_grads_per_tokenembed_BSM = feature_grads_per_tokenembed_BSM.view(n_games_total, model.cfg.n_ctx, model.cfg.d_model)

    # Approximate indirect effect of features on logits via attribution patching (zero ablation), grad * (patch - clean)
    ie_feature_to_logits_VBS = logit_grads_per_feature_VBS * (0 - feature_acts_BSF[..., feat_idx])
    # Each token occurs exactly once in the sequence, so we can sum over the sequence dim
    ie_feature_to_logits_VB = ie_feature_to_logits_VBS.sum(dim=-1)

    # Sort per token (instead of sequence pos)
    feature_acts_BS = feature_acts_BSF[..., feat_idx]
    acts_per_tokenembed_VB = t.zeros((model.cfg.d_vocab, n_games_total), device=device)
    for game_idx, game in enumerate(game_batches_NBS.view(n_games_total, -1)):
        acts_per_tokenembed_VB[game, game_idx] = feature_acts_BS[game_idx]

    if compute_ie_embed:
        # Approximate indirect effect of tokenembeds on features via attribution patching (mean ablation), grad * (patch - clean)
        mean_embed = embeds.mean(dim=(0,1))
        ie_tokenembed_to_features_BSM = feature_grads_per_tokenembed_BSM * (mean_embed - embeds)
        ie_tokenembed_to_features_BS = ie_tokenembed_to_features_BSM.sum(dim=-1)

        # Sort per token (instead of sequence pos)
        ie_tokenembed_to_features_VB = t.zeros((model.cfg.d_vocab, n_games_total), device=device) 
        for game_idx, game in enumerate(game_batches_NBS.view(n_games_total, -1)):
            ie_tokenembed_to_features_VB[game, game_idx] = ie_tokenembed_to_features_BS[game_idx]
            
        del feature_grads_per_tokenembed_BSM
        del ie_tokenembed_to_features_BS

    
    del feature_acts_BSF
    del embeds
    del logit_grads_per_feature_VBS
    del ie_feature_to_logits_VBS
    t.cuda.empty_cache()
    gc.collect()

    if compute_ie_embed:
        return feature_acts_BS, acts_per_tokenembed_VB, ie_feature_to_logits_VB, ie_tokenembed_to_features_VB
    return feature_acts_BS, acts_per_tokenembed_VB, ie_feature_to_logits_VB

def plot_mean_metrics(acts_per_tokenembed_VB, ie_tokenembed_to_features_VB, ie_feature_to_logits_VB, feat_idx, n_games, device):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    mean_feature_activations_tokenembed_V = acts_per_tokenembed_VB.mean(dim=-1)
    mean_ie_tokenembed_to_features_V = ie_tokenembed_to_features_VB.mean(dim=-1)
    mean_logit_attributions_V = ie_feature_to_logits_VB.mean(dim=-1)
    visualize_vocab(axs[0], mean_feature_activations_tokenembed_V[1:], title=f'Feature activations', device=device)
    visualize_vocab(axs[1], mean_ie_tokenembed_to_features_V[1:], title=f'IE token_embed --> feature\n(AP, mean ablation)', device=device)
    visualize_vocab(axs[2], mean_logit_attributions_V[1:], title=f'IE feature --> logits\n(AP, zero ablation)', device=device)

    fig.suptitle(f'Mean metrics over {n_games} games for feature #{feat_idx}')
    plt.show()

def plot_top_k_games(feat_idx, games_batch_BS, feature_acts_BS, acts_per_tokenembed_VB, ie_tokenembed_to_features_VB, ie_feature_to_logits_VB, device, sort_metric='activation', k=10):
    if sort_metric == 'activation':
        top_game_indices = acts_per_tokenembed_VB.abs().max(dim=0)[0].topk(k).indices
    elif sort_metric == 'ie_embed':
        top_game_indices = ie_tokenembed_to_features_VB.abs().max(dim=0)[0].topk(k).indices
    elif sort_metric == 'ie_logit':
        top_game_indices = ie_feature_to_logits_VB.abs().max(dim=0)[0].topk(k).indices

    print(top_game_indices)

    for i, game_idx in enumerate(top_game_indices):
        print(f"Top {i+1} {sort_metric} game:")

        game = games_batch_BS[game_idx].tolist()
        feature_act_S = feature_acts_BS[game_idx]
        max_abs_act = feature_act_S.abs().max()
        display(visualize_game_seq(game, feature_act_S, max_abs_act, prefix='feature activations: <br>'))

        acts_V = acts_per_tokenembed_VB[:, game_idx]
        ie_embed_V = ie_tokenembed_to_features_VB[:, game_idx]
        ie_logit_V = ie_feature_to_logits_VB[:, game_idx]


        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        visualize_vocab(axs[0], acts_V[1:], title=f'Feature activations', device=device)
        visualize_vocab(axs[1], ie_embed_V[1:], title=f'IE token_embed --> feature\n(AP, mean ablation)', device=device)
        visualize_vocab(axs[2], ie_logit_V[1:], title=f'IE feature --> logits\n(AP, zero ablation)', device=device)
        fig.suptitle(f'Top {i+1} {sort_metric} game for feature #{feat_idx}')
        plt.show()