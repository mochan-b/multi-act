# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, eval_num_queries, camera_names, qpos_history, action_history, image_history=None, history_backbones=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.eval_num_queries = eval_num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        
        # History parameters
        self.qpos_history = qpos_history
        self.qpos_history_max = max(qpos_history) if qpos_history else 0
        self.qpos_history_len = len(qpos_history) if qpos_history else 0
        
        self.action_history = action_history
        self.action_history_max = max(action_history) if action_history else 0
        self.action_history_len = len(action_history) if action_history else 0
        
        self.image_history = image_history if image_history else []
        self.image_history_max = max(image_history) if image_history else 0
        self.image_history_len = len(image_history) if image_history else 0

        # Position embeddings for all history (qpos and action). We will have a sinusoidal position embedding
        # and then a order tag to differentiate between the different positions in the history with
        # the sinusoidal position embedding for the main sequence.
        # Fixed sinusoidal embedding → register as buffer
        history_max = max(self.qpos_history_max, self.action_history_max, self.image_history_max) if (self.qpos_history_max > 0 or self.action_history_max > 0 or self.image_history_max > 0) else 0
        if history_max > 0:
            self.register_buffer('history_pos_embed', get_sinusoid_encoding_table(history_max + 1, hidden_dim))
        # Order tag → learnable (trainable)
        self.qpos_history_order_tag_enc = nn.Parameter(torch.randn(1, 1, hidden_dim)) # learnable order tag for encoder
        self.qpos_history_order_tag_dec = nn.Parameter(torch.randn(1, 1, hidden_dim)) # learnable order tag for decoder
        self.action_history_order_tag_enc = nn.Parameter(torch.randn(1, 1, hidden_dim)) # learnable order tag for encoder
        self.action_history_order_tag_dec = nn.Parameter(torch.randn(1, 1, hidden_dim)) # learnable order tag for decoder
        self.image_history_order_tag_enc = nn.Parameter(torch.randn(1, 1, hidden_dim)) # learnable order tag for encoder
        self.image_history_order_tag_dec = nn.Parameter(torch.randn(1, 1, hidden_dim)) # learnable order tag for decoder

        # Make an encoder for each of the qpos history positions
        self.qpos_history_enc = nn.ModuleList([
            nn.Linear(14, hidden_dim) for _ in range(self.qpos_history_len)
        ]) # For CVAE encoder
        self.qpos_history_dec = nn.ModuleList([
            nn.Linear(14, hidden_dim) for _ in range(self.qpos_history_len)
        ]) # For CVAE decoder
        
        # Make an encoder for each of the action history positions
        self.action_history_enc = nn.ModuleList([
            nn.Linear(14, hidden_dim) for _ in range(self.action_history_len)
        ]) # For CVAE encoder
        self.action_history_dec = nn.ModuleList([
            nn.Linear(14, hidden_dim) for _ in range(self.action_history_len)
        ]) # For CVAE decoder

        # CVAE encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim)  # project action to embedding for CVAE
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding for CVAE

        # Encoders for the decoder
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            
            # History backbones for image history processing (decoder only)
            if history_backbones is not None and self.image_history_len > 0:
                self.history_backbones = nn.ModuleList([
                    nn.ModuleList(history_backbones) for _ in range(self.image_history_len)
                ])
                # Input projection for history images (might have different channels)
                self.history_input_proj = nn.ModuleList([
                    nn.Conv2d(history_backbones[0].num_channels, hidden_dim, kernel_size=1) 
                    for _ in range(self.image_history_len)
                ])
            else:
                self.history_backbones = None
                self.history_input_proj = None
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim) # Encoders for the joints 
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # Multi-ACT model extra parameters
        self.input_task_embeddings = nn.Linear(512, hidden_dim)
        self.camera_embeddings = nn.Embedding(len(camera_names), hidden_dim)
        
        self.encoder_task_proj = nn.Linear(512, hidden_dim)  # project task embedding to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        
        # Calculate total sequence length for positional embeddings
        # [CLS], qpos, task, action_sequence, qpos_history, action_history
        pos_table_size = 1 + 1 + 1 + num_queries + self.qpos_history_len + self.action_history_len
        self.register_buffer('pos_table', get_sinusoid_encoding_table(pos_table_size, hidden_dim))

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        additional_pos_embed_size = 3 + self.qpos_history_len + self.action_history_len + self.image_history_len  # latent_input, proprio_input, task_name_input, qpos_history, action_history, image_history
        self.additional_pos_embed = nn.Embedding(additional_pos_embed_size, hidden_dim)

    def forward(self, qpos_data, image_data, env_state, actions=None, action_history_data=None, task_embeddings=None, camera_indices=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        
        # Take qpos as the first element (current timestep)
        qpos = qpos_data[:, 0]
        # Take qpos history as everything after the first element
        qpos_history = qpos_data[:, 1:]
        
        # Image data with history. First element is the current value
        image = image_data[:, 0]
        image_history = image_data[:, 1:]

        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)

            # Use qpos history.
            qpos_history_fixed_embed = None
            if qpos_history.shape[1] > 0:
                history_length = qpos_history.shape[1]
                qpos_history_embeds = []
                
                for i in range(history_length):
                    # Get the qpos at this history position
                    qpos_at_i = qpos_history[:, i]  # (bs, qpos_dim)
                    
                    # Project through encoder for this history position
                    qpos_history_embed = self.qpos_history_enc[i](qpos_at_i)  # (bs, hidden_dim)
                    qpos_history_embed = torch.unsqueeze(qpos_history_embed, axis=1)  # (bs, 1, hidden_dim)
                    
                    # Add positional embedding (history position i gets position embedding i)
                    qpos_idx = self.qpos_history[i]  # Get actual history index
                    pos_embed_for_qpos_history = self.history_pos_embed[:, qpos_idx:qpos_idx+1, :]  # (1, 1, hidden_dim)
                    qpos_history_embed = qpos_history_embed + pos_embed_for_qpos_history
                    
                    # Add order tag
                    qpos_history_embed = qpos_history_embed + self.qpos_history_order_tag_enc
                    
                    qpos_history_embeds.append(qpos_history_embed)
                
                # Concatenate all history embeddings
                qpos_history_fixed_embed = torch.cat(qpos_history_embeds, axis=1)  # (bs, history_length, hidden_dim)

            # Use action history.
            action_history_fixed_embed = None
            if action_history_data is not None and action_history_data.shape[1] > 0:
                history_length = action_history_data.shape[1]
                action_history_embeds = []
                
                for i in range(history_length):
                    # Get the action at this history position
                    action_at_i = action_history_data[:, i]  # (bs, action_dim)
                    
                    # Project through encoder for this history position
                    action_history_embed = self.action_history_enc[i](action_at_i)  # (bs, hidden_dim)
                    action_history_embed = torch.unsqueeze(action_history_embed, axis=1)  # (bs, 1, hidden_dim)
                    
                    # Add positional embedding (use actual history index, not loop index)
                    history_idx = self.action_history[i]  # Get actual history index like 3, 10
                    pos_embed_for_action_history = self.history_pos_embed[:, history_idx:history_idx+1, :]  # (1, 1, hidden_dim)
                    action_history_embed = action_history_embed + pos_embed_for_action_history
                    
                    # Add order tag
                    action_history_embed = action_history_embed + self.action_history_order_tag_enc
                    
                    action_history_embeds.append(action_history_embed)
                
                # Concatenate all history embeddings
                action_history_fixed_embed = torch.cat(action_history_embeds, axis=1)  # (bs, history_length, hidden_dim)

            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            task_name_embed = self.encoder_task_proj(task_embeddings)  # (bs, hidden_dim)
            task_name_embed = torch.unsqueeze(task_name_embed, axis=1)
            # Build encoder input with all available components
            input_components = [cls_embed, qpos_embed, task_name_embed]
            if qpos_history_fixed_embed is not None:
                input_components.append(qpos_history_fixed_embed)
            if action_history_fixed_embed is not None:
                input_components.append(action_history_fixed_embed)
            input_components.append(action_embed)
            
            encoder_input = torch.cat(input_components, axis=1)  # (bs, seq_len, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token and fixed components
            fixed_components_len = 3 + self.qpos_history_len + self.action_history_len  # cls + qpos + task + qpos_history + action_history
            cls_joint_is_pad = torch.full((bs, fixed_components_len), False).to(qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, total_seq_len)
            # obtain position embedding - now fixed components first, then variable actions  
            # New order: [cls, qpos, task, qpos_history, action_history, actions]
            pos_embed_for_qpos = self.pos_table.clone().detach()
            pos_embed_for_qpos = pos_embed_for_qpos.permute(1, 0, 2)  # (total_size, 1, hidden_dim)
            
            # Calculate sequence length: fixed components + variable action length
            fixed_len = 1 + 1 + 1 + self.qpos_history_len + self.action_history_len  # cls + qpos + task + qpos_history + action_history
            total_seq_len = fixed_len + actions.shape[1]  # fixed + variable actions
            pos_embed_for_qpos = pos_embed_for_qpos[:total_seq_len]
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed_for_qpos, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Process qpos history for decoder (conditional input to transformer)
        qpos_history_decoder_embed = None
        if qpos_history.shape[1] > 0:
            history_length = qpos_history.shape[1]
            qpos_history_decoder_embeds = []
            
            for i in range(history_length):
                # Get the qpos at this history position
                qpos_at_i = qpos_history[:, i]  # (bs, qpos_dim)
                
                # Project through decoder encoder for this history position
                qpos_history_embed = self.qpos_history_dec[i](qpos_at_i)  # (bs, hidden_dim)
                qpos_history_embed = torch.unsqueeze(qpos_history_embed, axis=1)  # (bs, 1, hidden_dim)
                
                # Add positional embedding (history position i gets position embedding i)
                qpos_idx = self.qpos_history[i]  # Get actual history index
                pos_embed_for_qpos = self.history_pos_embed[:, qpos_idx:qpos_idx+1, :]  # (1, 1, hidden_dim)
                qpos_history_embed = qpos_history_embed + pos_embed_for_qpos
                
                # Add order tag for decoder
                qpos_history_embed = qpos_history_embed + self.qpos_history_order_tag_dec
                
                qpos_history_decoder_embeds.append(qpos_history_embed)
            
            # Concatenate all history embeddings
            qpos_history_decoder_embed = torch.cat(qpos_history_decoder_embeds, axis=1)  # (bs, history_length, hidden_dim)

        # Process action history for decoder (conditional input to transformer)
        action_history_decoder_embed = None
        if action_history_data is not None and action_history_data.shape[1] > 0:
            history_length = action_history_data.shape[1]
            action_history_decoder_embeds = []
            
            for i in range(history_length):
                # Get the action at this history position
                action_at_i = action_history_data[:, i]  # (bs, action_dim)
                
                # Project through decoder encoder for this history position
                action_history_embed = self.action_history_dec[i](action_at_i)  # (bs, hidden_dim)
                action_history_embed = torch.unsqueeze(action_history_embed, axis=1)  # (bs, 1, hidden_dim)
                
                # Add positional embedding (use actual history index, not loop index)
                history_idx = self.action_history[i]  # Get actual history index like 3, 10
                pos_embed = self.history_pos_embed[:, history_idx:history_idx+1, :]  # (1, 1, hidden_dim)
                action_history_embed = action_history_embed + pos_embed
                
                # Add order tag for decoder
                action_history_embed = action_history_embed + self.action_history_order_tag_dec
                
                action_history_decoder_embeds.append(action_history_embed)
            
            # Concatenate all history embeddings
            action_history_decoder_embed = torch.cat(action_history_decoder_embeds, axis=1)  # (bs, history_length, hidden_dim)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []

            # Pass the images through their respective backbones
            image = image.permute(1, 0, 2, 3, 4)
            permuted_backbones = [self.backbones[index] for index in camera_indices]
            output = [backbone(x_) for backbone, x_ in zip(permuted_backbones, image)]

            # Get the features embeddings for each camera
            cam_embeddings = [self.camera_embeddings(torch.tensor([index]).to(image.device)).view(1, 512, 1, 1) for
                              index in camera_indices]

            # Concatenate the features and the camera embeddings
            all_cam_features = [self.input_proj(features[0]) + cam_embeddings[index] for index, (features, _) in
                                enumerate(output)]
            all_cam_pos = [pos[0] for _, pos in output]

            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            task_name_input = self.input_task_embeddings(task_embeddings)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # Make the query embed match the shape of the input action sequence
            if is_training:                
                query_embed = self.query_embed.weight[:actions.shape[1]]
            else:
                query_embed = self.query_embed.weight[:self.eval_num_queries]

            # Process image history
            image_history_embed = None
            if self.history_backbones is not None and image_history.shape[1] > 0:
                history_length = image_history.shape[1]
                image_history_embeds = []
                
                for i in range(history_length):
                    # Get the image at this history position
                    hist_image = image_history[:, i]  # (bs, num_cam, H, W, C)
                    
                    # Process through history backbone for this position
                    hist_image = hist_image.permute(1, 0, 2, 3, 4)  # (num_cam, bs, H, W, C)
                    permuted_history_backbones = [self.history_backbones[i][index] for index in camera_indices]
                    hist_output = [backbone(x_) for backbone, x_ in zip(permuted_history_backbones, hist_image)]
                    
                    # Get camera embeddings (same as current images)
                    hist_cam_embeddings = [self.camera_embeddings(torch.tensor([index]).to(hist_image.device)).view(1, 512, 1, 1) 
                                          for index in camera_indices]
                    
                    # Project and add camera embeddings
                    hist_cam_features = [self.history_input_proj[i](features[0]) + hist_cam_embeddings[idx] 
                                        for idx, (features, _) in enumerate(hist_output)]
                    
                    # Fold camera dimension into width dimension
                    hist_src = torch.cat(hist_cam_features, axis=3)  # (bs, hidden_dim, H, total_W)
                    
                    # Global average pooling to get fixed-size representation
                    hist_embed = torch.mean(hist_src, dim=[2, 3])  # (bs, hidden_dim)
                    hist_embed = torch.unsqueeze(hist_embed, axis=1)  # (bs, 1, hidden_dim)
                    
                    # Add positional embedding
                    hist_idx = self.image_history[i]  # Get actual history index
                    pos_embed_for_image_history = self.history_pos_embed[:, hist_idx:hist_idx+1, :]  # (1, 1, hidden_dim)
                    hist_embed = hist_embed + pos_embed_for_image_history
                    
                    # Add order tag
                    hist_embed = hist_embed + self.image_history_order_tag_dec
                    
                    image_history_embeds.append(hist_embed)
                
                # Concatenate all image history embeddings
                image_history_embed = torch.cat(image_history_embeds, axis=1)  # (bs, history_length, hidden_dim)

            # Combine qpos, action, and image history for transformer
            combined_history_embed = None
            history_components = []
            if qpos_history_decoder_embed is not None:
                history_components.append(qpos_history_decoder_embed)
            if action_history_decoder_embed is not None:
                history_components.append(action_history_decoder_embed)
            if image_history_embed is not None:
                history_components.append(image_history_embed)
            
            if history_components:
                combined_history_embed = torch.cat(history_components, axis=1)
            
            hs = self.transformer(src, None, query_embed, pos, latent_input, proprio_input, task_name_input,
                                  self.additional_pos_embed.weight, combined_history_embed)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)  # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]  # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, qpos], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14  # TODO hardcode

    # Build backbone for each camera
    backbones = [build_backbone(args) for _ in args.camera_names]

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    # Pass the qpos_history, action_history, and image_history only if multi_history is enabled
    if args.multi_history:
        qpos_history = args.qpos_history
        action_history = args.action_history
        image_history = getattr(args, 'image_history', [])
    else:
        qpos_history = []
        action_history = []
        image_history = []

    # Build history backbones if image_history is enabled
    history_backbones = None
    if image_history and hasattr(args, 'history_backbone'):
        history_backbones = [build_backbone(args, args.history_backbone) for _ in args.camera_names]

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        eval_num_queries=args.eval_num_queries,
        camera_names=args.camera_names,
        qpos_history=qpos_history,
        action_history=action_history,
        image_history=image_history,
        history_backbones=history_backbones,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_cnnmlp(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
