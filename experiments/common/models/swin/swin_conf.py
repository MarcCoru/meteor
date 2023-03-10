# parsed from https://github.com/HSG-AIML/SSLTransformerRS/blob/main/configs/backbone_config.json

class dotdict(dict):
 """dot.notation access to dictionary attributes"""

 __getattr__ = dict.get
 __setattr__ = dict.__setitem__
 __delattr__ = dict.__delitem__


def dotdictify(d):
 """recursively wrap a dictionary and
 all the dictionaries that it contains
 with the dotdict class
 """
 d = dotdict(d)
 for k, v in d.items():
  if isinstance(v, dict):
   d[k] = dotdictify(v)
 return d

swin_conf_dict = {'epochs': 301,
 'batch_size': 50,
 'seed': 42,
 'dataloader_workers': 8,
 'train_dir': '/ds2/remote_sensing/sen12ms',
 'train_mode': 'sen12ms',
 'val_dir': '/netscratch/lscheibenreif/grss-dfc-20',
 'val_mode': 'validation',
 'clip_sample_values': True,
 'transforms': None,
 'used_data_fraction': 1,
 's1_input_channels': 2,
 's2_input_channels': 13,
 'image_px_size': 224,
 'cover_all_parts_validation': True,
 'cover_all_parts_train': False,
 'balanced_classes_train': False,
 'balanced_classes_validation': False,
 'target': 'dfc_label',
 'model_config_path': 'Transformer_SSL/configs/moby_swin_tiny.yaml',
 'fp16_precision': True,
 'out_dim': 128,
 'n_views': 2,
 'device': 'cuda',
 'disable_cuda': False,
 'log_every_n_steps': 10000,
 'use_logging': False,
 'model_config': {'TRAIN': {'WARMUP_EPOCHS': 5,
   'EPOCHS': 300,
   'BASE_LR': 0.001,
   'WEIGHT_DECAY': 0.05},
  'AUG': {'SSL_AUG': True},
  'MODEL': {'TYPE': 'd-swin',
   'NAME': 'moby__swin_tiny__patch4_window7_126__odpr02_tdpr0_cm099_ct02_queue4096_proj2_pred2',
   'SWIN': {'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 6, 2],
    'NUM_HEADS': [3, 6, 12, 24],
    'WINDOW_SIZE': 7,
    'PATCH_SIZE': 4,
    'IN_CHANS': 2,
    'MLP_RATIO': 4.0,
    'QKV_BIAS': True,
    'QK_SCALE': None,
    'APE': False,
    'PATCH_NORM': 'ln'},
   'MOBY': {'ENCODER': 'swin',
    'ONLINE_DROP_PATH_RATE': 0.2,
    'TARGET_DROP_PATH_RATE': 0.0,
    'CONTRAST_MOMENTUM': 0.99,
    'CONTRAST_TEMPERATURE': 0.2,
    'CONTRAST_NUM_NEGATIVE': 4096,
    'PROJ_NUM_LAYERS': 2,
    'PRED_NUM_LAYERS': 2},
   'DROP_RATE': 0.0,
   'DROP_PATH_RATE': 0.1,
   'NUM_CLASSES': 1000,
   'TRAIN': {'USE_CHECKPIONT': False, 'TRAINING_IMAGES': 1000}},
  'DATA': {'IMG_SIZE': 224}},
 'TRAIN': {'OPTIMIZER': {'NAME': 'AdamW',
   'MOMENTUM': 0.9,
   'EPS': 1e-08,
   'BETAS': [0.9, 0.999]},
  'LR_SCHEDULER': {'NAME': 'cosine', 'DECAY_EPOCHS': 30, 'DECAY_RATE': 0.1},
  'BASE_LR': 0.0005,
  'WARMUP_LR': 5e-07,
  'MIN_LR': 5e-06,
  'CLIP_GRAD': 5.0,
  'AUTP_RESUME': True,
  'ACCUMULATION_STEPS': 0,
  'USE_CHECKPOINT': False,
  'WEIGHT_DECAY': 0,
  'START_EPOCH': 0,
  'EPOCHS': 300,
  'WARMUP_EPOCHS': 20,
  'CONTRAST_TEMPERATURE': 0.2},
 'AMP_OPT_LEVEL': 'O0',
 'OUTPUT': 'checkpoints/'}


swin_conf = dotdictify(swin_conf_dict)
