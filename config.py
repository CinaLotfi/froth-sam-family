from pathlib import Path


ROOT = Path(__file__).parent


class Config:
    # ---------- General ----------
    seed: int = 1337
    device: str = "auto"          # "cuda", "cpu", or "auto"

    # ---------- Data ----------
    data_root = ROOT / "data"
    train_dir = data_root / "train"
    test_dir  = data_root / "test"
    label_key: str = "froth"      # Label name in LabelMe JSON

    # ---------- Model selection ----------
    # "sam" | "medsam" | "hqsam"
    model_name: str = "sam"

    # ---------- Checkpoints ----------
    weights_dir = ROOT / "weights"

    # official / base weights for each model
    sam_checkpoint   = weights_dir / "sam_vit_b_01ec64.pth"
    medsam_checkpoint = weights_dir / "medsam.pth"      # to be added later
    hqsam_checkpoint  = weights_dir / "hqsam.pth"       # to be added later

    # model types (for SAM registry etc.)
    sam_model_type: str = "vit_b"
    # medsam/hqsam may have their own types; we'll handle that later

    # ---------- Training mode ----------
    # one of: "decoder_only" | "decoder+prompt" | "prompt_only" | "encoder_only" | "full"
    train_mode: str = "full"

    # ---------- Dataloader ----------
    batch_size: int = 1           # keep 1, we use gradient accumulation
    num_workers: int = 0
    pin_memory: bool = True

    # ---------- Training hyperparameters ----------
    epochs: int = 10

    # per-module learning rates (used for SAM-like models)
    lr_decoder: float = 1e-5
    lr_prompt: float  = 5e-6
    lr_encoder: float = 1e-6
    weight_decay: float = 0.0

    # gradient accumulation
    accum_steps: int = 8

    # mixed precision
    use_amp: bool = True

    # gradient clipping
    grad_clip_norm: float | None = 1.0

    # optional warm-start decoder from previous finetune
    load_decoder_from = None

    # ---------- Outputs ----------
    outputs_root = ROOT / "outputs"
    finetune_out = outputs_root / "sam_finetune"

    @classmethod
    def setup(cls):
        cls.outputs_root.mkdir(parents=True, exist_ok=True)
        cls.finetune_out.mkdir(parents=True, exist_ok=True)

        if not cls.sam_checkpoint.exists():
            print(
                f"[WARN] SAM checkpoint not found at {cls.sam_checkpoint}. "
                "Download sam_vit_b_01ec64.pth into weights/ if you want to use 'sam'."
            )

        if not cls.train_dir.exists():
            print(f"[WARN] Train dir does not exist yet: {cls.train_dir}")
        if not cls.test_dir.exists():
            print(f"[WARN] Test dir does not exist yet:  {cls.test_dir}")
