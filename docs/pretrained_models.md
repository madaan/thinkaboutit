# Pre-trained models

- The pre-trained models for the three domains with clean and noisy graphs are located [here](https://drive.google.com/file/d/1QKSnMLpt0TfM-Jxu-eI-c92qHSjcIAov/view?usp=sharing)
```
pre_trained_models
├── corrected_graphs
│   ├── gcn
│   │   ├── atomic
│   │   │   └── epoch=28-step=31178-val_acc_epoch=0.7857.ckpt
│   │   ├── snli
│   │   │   └── epoch=24-step=69298-val_acc_epoch=0.8398.ckpt
│   │   └── social
│   │       └── epoch=28-step=68599-val_acc_epoch=0.8795.ckpt
│   ├── gcn_moe
│   │   ├── atomic
│   │   │   └── epoch=27-step=30631-val_acc_epoch=0.7914.ckpt
│   │   ├── snli
│   │   │   └── epoch=21-step=59597-val_acc_epoch=0.8364.ckpt
│   │   └── social
│   │       └── epoch=28-step=69802-val_acc_epoch=0.8771.ckpt
│   ├── moe
│   │   ├── atomic
│   │   │   └── epoch=28-step=31178-val_acc_epoch=0.7995.ckpt
│   │   ├── snli
│   │   │   └── epoch=27-step=76229-val_acc_epoch=0.8448.ckpt
│   │   └── social
│   │       └── epoch=29-step=72209-val_acc_epoch=0.8824.ckpt
│   └── str
│       ├── atomic
│       │   └── epoch=13-step=14768-val_acc_epoch=0.7971.ckpt
│       ├── snli
│       │   └── epoch=14-step=40193-val_acc_epoch=0.8471.ckpt
│       └── social
│           └── epoch=4-step=10830-val_acc_epoch=0.8635.ckpt
└── original_graphs
    ├── gcn
    │   ├── atomic
    │   │   └── epoch=19-step=21879-val_acc_epoch=0.7833.ckpt
    │   ├── snli
    │   │   └── epoch=23-step=65141-val_acc_epoch=0.8398.ckpt
    │   └── social
    │       └── epoch=25-step=61378-val_acc_epoch=0.8785.ckpt
    ├── moe
    │   ├── atomic
    │   │   └── epoch=4-step=5469-val_acc_epoch=0.7956.ckpt
    │   ├── snli
    │   │   └── epoch=18-step=52666-val_acc_epoch=0.8426.ckpt
    │   └── social
    │       └── epoch=28-step=69802-val_acc_epoch=0.8819.ckpt
    └── str
        ├── atomic
        │   └── epoch=13-step=14768-val_acc_epoch=0.7971.ckpt
        ├── snli
        │   └── epoch=7-step=22174-val_acc_epoch=0.8415.ckpt
        └── social
            └── epoch=5-step=13237-val_acc_epoch=0.8712.ckpt
```