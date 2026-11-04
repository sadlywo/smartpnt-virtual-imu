"""Model factory and model exports."""

from __future__ import annotations

from typing import Any


def make_model(config: dict) -> Any:
    """Create a model from a config dictionary."""
    model_type = str(config.get("type", "tcn")).lower()
    if model_type == "tcn":
        from virtual_imu.models.tcn import TCN

        return TCN(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 48)),
            levels=int(config.get("levels", 4)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
            zero_init_head=bool(config.get("zero_init_head", False)),
        )
    if model_type in {"film_tcn", "conditional_tcn"}:
        from virtual_imu.models.tcn import FiLMTCN

        return FiLMTCN(
            in_channels=int(config.get("in_channels", 12)),
            out_channels=int(config.get("out_channels", 6)),
            context_channels=int(config.get("context_channels", max(1, int(config.get("in_channels", 12)) - 7))),
            hidden_channels=int(config.get("hidden_channels", 48)),
            levels=int(config.get("levels", 4)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
            zero_init_head=bool(config.get("zero_init_head", False)),
        )
    if model_type in {"bias_drift_tcn", "two_branch_tcn"}:
        from virtual_imu.models.tcn import BiasDriftTCN

        return BiasDriftTCN(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            static_channels=int(config.get("static_channels", 7)),
            hidden_channels=int(config.get("hidden_channels", 48)),
            levels=int(config.get("levels", 4)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
            use_bias_branch=bool(config.get("use_bias_branch", True)),
            use_drift_branch=bool(config.get("use_drift_branch", True)),
            bias_branch_mode=str(config.get("bias_branch_mode", "offset")),
            branch_init_scale=float(config.get("branch_init_scale", 0.1)),
            zero_init_head=bool(config.get("zero_init_head", False)),
        )
    if model_type == "gru":
        from virtual_imu.models.gru import GRUModel

        return GRUModel(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if model_type == "transformer":
        from virtual_imu.models.transformer import TransformerIMU

        return TransformerIMU(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            d_model=int(config.get("d_model", 64)),
            nhead=int(config.get("nhead", 4)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            dim_feedforward=int(config.get("dim_feedforward", int(config.get("d_model", 64)) * int(config["ff_multiplier"]) if "ff_multiplier" in config else 2048)),
            norm_first=bool(config.get("norm_first", False)),
            activation=str(config.get("activation", "relu")),
        )
    if model_type in {"informer", "patchtst", "itransformer"}:
        from virtual_imu.models.time_series_attention import (
            InformerIMU,
            ITransformerIMU,
            PatchTSTIMU,
        )

        common = {
            "in_channels": int(config.get("in_channels", 6)),
            "out_channels": int(config.get("out_channels", 6)),
            "window_size": int(config.get("window_size", 80)),
            "d_model": int(config.get("d_model", 64)),
            "nhead": int(config.get("nhead", 4)),
            "num_layers": int(config.get("num_layers", 2)),
            "dim_feedforward": int(config.get("dim_feedforward", int(config.get("d_model", 64)) * int(config.get("ff_multiplier", 2)))),
            "dropout": float(config.get("dropout", 0.1)),
        }
        if model_type == "informer":
            return InformerIMU(**common, factor=int(config.get("factor", 5)), distil=bool(config.get("distil", True)), sampling_seed=int(config.get("sampling_seed", 0)))
        if model_type == "patchtst":
            return PatchTSTIMU(**common, patch_len=int(config.get("patch_len", 16)), stride=int(config.get("stride", 8)))
        return ITransformerIMU(**common)
    if model_type in {"unet", "unet1d"}:
        from virtual_imu.models.unet1d import UNet1D

        return UNet1D(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            base_channels=int(config.get("base_channels", 32)),
        )
    if model_type in {"brossard", "brossard_cnn", "dilated_cnn"}:
        from virtual_imu.models.advanced import BrossardDilatedCNN

        return BrossardDilatedCNN(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 64)),
            levels=int(config.get("levels", 5)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if model_type in {"calibnet", "calibnet_cnn", "plain_cnn"}:
        from virtual_imu.models.advanced import CalibNetCNN

        return CalibNetCNN(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 64)),
            kernel_size=int(config.get("kernel_size", 7)),
            depth=int(config.get("depth", 4)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if model_type in {"ronin", "ronin_resnet", "ronin_resnet_adapter"}:
        from virtual_imu.models.advanced import RoNINResNet

        return RoNINResNet(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 64)),
            blocks=int(config.get("blocks", 5)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if model_type in {"alimu", "alimu_cnn_bilstm", "alimu_mapper", "cnn_bilstm"}:
        from virtual_imu.models.advanced import ALiMUCNNBiLSTM

        return ALiMUCNNBiLSTM(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 64)),
            lstm_hidden=int(config.get("lstm_hidden", 64)),
            conv_depth=int(config.get("conv_depth", 2)),
            lstm_layers=int(config.get("lstm_layers", 1)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
            bidirectional=bool(config.get("bidirectional", True)),
        )
    if model_type in {"sixdof_cnn_bilstm", "imu_6dof_cnn_bilstm"}:
        from virtual_imu.models.advanced import SixDOFCNNBiLSTM

        return SixDOFCNNBiLSTM(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 64)),
            lstm_hidden=int(config.get("lstm_hidden", 128)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if model_type in {"fsda", "fsda_denoiser", "domain_adaptation_denoiser"}:
        from virtual_imu.models.advanced import FSDADenoiser

        return FSDADenoiser(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 64)),
            latent_channels=int(config.get("latent_channels", 48)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if model_type == "deepgyro_backbone":
        from virtual_imu.models.published_backbones import DeepGyroBackboneAdapter

        return DeepGyroBackboneAdapter(
            in_channels=int(config.get("in_channels", 6)),
            out_channels=int(config.get("out_channels", 3)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_type == "tlio_resnet18":
        from virtual_imu.models.published_backbones import ResNet18SequenceAdapter

        return ResNet18SequenceAdapter(
            in_channels=int(config.get("in_channels", 6)),
            out_channels=int(config.get("out_channels", 3)),
        )
    if model_type == "tartanimu_window":
        from virtual_imu.models.tartan_window import TartanIMUWindowAdapter

        return TartanIMUWindowAdapter(
            in_channels=int(config.get("in_channels", 6)),
            out_channels=int(config.get("out_channels", 3)),
            window_size=int(config.get("window_size", 80)),
            lstm_hidden=int(config.get("lstm_hidden", 128)),
            lstm_layers=int(config.get("lstm_layers", 2)),
            lstm_dropout=float(config.get("lstm_dropout", 0.1)),
            head_dropout=float(config.get("head_dropout", 0.3)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")
