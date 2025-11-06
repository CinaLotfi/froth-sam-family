from pathlib import Path


class Config:
    """
    Global config for froth-sam-family.
    Works for:
      - SAM      (model_name = "sam")
      - HQ-SAM   (model_name = "hqsam")
      - MedSAM   (model_name = "medsam")
    """

    # --------- paths ----------
    project_root = Path(__file__).resolve().parent

    data_root = project_root / "data"
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    eval_dir = data_root / "eval"   # optional

    outputs_root = project_root / "outputs"
    weights_root = project_root / "weights"

    # --------- dataset ----------
    label_key = "froth"

    # --------- training basics ----------
    seed = 1337
    device = "auto"        # "cuda" | "cpu" | "auto"
    batch_size = 1         # keep 1 for SAM / HQ-SAM / MedSAM box prompts
    num_workers = 0
    pin_memory = True

    # --------- optimization ----------
    epochs = 10
    lr = 1e-5
    weight_decay = 0.0

    # --------- model selection ----------
    # will be set via CLI: --model sam / --model hqsam / --model medsam
    model_name = "sam"            # "sam" | "hqsam" | "medsam"
    train_mode = "decoder_only"   # just used in filenames

    # SAM
    sam_checkpoint   = weights_root / "sam_vit_b_01ec64.pth"
    sam_model_type   = "vit_b"

    # HQ-SAM
    hqsam_checkpoint = weights_root / "sam_hq_vit_b.pth"
    hqsam_model_type = "vit_b"

    # MedSAM (NEW)
    medsam_checkpoint = weights_root / "medsam_vit_b.pth"
    medsam_model_type = "vit_b"

    # set in setup()
    finetune_out = outputs_root / "sam_finetune_out"

    @classmethod
    def setup(cls):
        """Make sure dirs exist and pick finetune output dir based on model."""
        cls.outputs_root.mkdir(parents=True, exist_ok=True)
        cls.weights_root.mkdir(parents=True, exist_ok=True)
        cls.data_root.mkdir(parents=True, exist_ok=True)

        if cls.model_name.lower() == "sam":
            subdir = "sam_finetune_out"
        elif cls.model_name.lower() == "hqsam":
            subdir = "hqsam_finetune_out"
        else:
            # e.g. "medsam_finetune_out"
            subdir = f"{cls.model_name}_finetune_out"

        cls.finetune_out = cls.outputs_root / subdir
        cls.finetune_out.mkdir(parents=True, exist_ok=True)
