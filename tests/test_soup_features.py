import json
import pytest
from pathlib import Path

from myai.training.layer_streaming import LayerStreamingManager, LayerStreamingConfig
from myai.training.preference_losses import dpo_loss, orpo_loss, simpo_loss, kto_loss
from myai.training.dataset_builder import extract_preference_pairs
from myai.evaluation.reward_synth import infer_verifier, calibrate_verifier, synthesize_reward_script
from myai.evaluation.ship_gate import run_ship_gate
from myai.export.merger import merge_adapter
from myai.export.gguf_exporter import export_to_gguf
from myai.hardware.feasibility import TrainingConfig, estimate_vram_gb, check_feasibility


def test_layer_streaming_vram_reduction():
    """Verify that layer streaming reduces peak estimated VRAM to ~3.32 GB for 8B models."""
    model_mock = {"params_b": 8.0, "num_layers": 32, "hidden_size": 4096}
    
    # Standard 4-bit resident training
    standard_cfg = TrainingConfig(quantization="4bit", stream_layers=False)
    standard_vram = estimate_vram_gb(model_mock, standard_cfg)
    assert standard_vram > 5.5, f"Standard 8B VRAM should be > 5.5 GB, got {standard_vram}"

    # Layer streaming enabled
    stream_cfg = TrainingConfig(quantization="4bit", stream_layers=True)
    stream_vram = estimate_vram_gb(model_mock, stream_cfg)
    assert stream_vram < 3.8, f"Layer streaming 8B VRAM should be < 3.8 GB, got {stream_vram}"
    assert stream_vram <= 3.5, f"Peak VRAM is capped to ~3.3-3.5 GB, got {stream_vram}"

    # Verify manager methods
    mgr = LayerStreamingManager(LayerStreamingConfig(enabled=True))
    mgr.initialize_store(32)
    assert mgr.telemetry.total_layers == 32

    fwd = mgr.stream_forward_layer(0)
    assert fwd["status"] == "in_vram"
    assert mgr.telemetry.active_layers_in_vram == 1

    mgr.offload_layer(0)
    assert mgr.telemetry.active_layers_in_vram == 0


def test_feasibility_auto_activates_layer_streaming():
    """Verify check_feasibility activates layer streaming on a 4GB GPU for an 8B model when permitted."""
    hw_4gb = type("HW", (), {"has_gpu": True, "vram_gb": 4.0, "ram_gb": 16.0})()
    model_8b = {"params_b": 8.0, "num_layers": 32, "hidden_size": 4096}

    report = check_feasibility(hw_4gb, model_8b, allow_layer_streaming=True)
    assert report.is_feasible is True
    assert report.recommended_config.stream_layers is True
    assert any("Layer Streaming" in r for r in report.reasons)


def test_preference_losses():
    """Verify DPO, ORPO, SimPO, and KTO preference loss functions."""
    policy_chosen = [2.5, 3.0, 1.8]
    policy_rejected = [1.0, 1.2, 0.5]
    ref_chosen = [2.0, 2.8, 1.5]
    ref_rejected = [1.1, 1.5, 0.8]
    len_chosen = [20, 25, 18]
    len_rejected = [15, 18, 12]

    # DPO
    loss_dpo, metrics_dpo = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert loss_dpo > 0.0
    assert "accuracy" in metrics_dpo
    assert metrics_dpo["accuracy"] == 1.0

    # SimPO
    loss_simpo, metrics_simpo = simpo_loss(policy_chosen, policy_rejected, len_chosen, len_rejected, beta=2.0, gamma=0.5)
    assert loss_simpo > 0.0
    assert "accuracy" in metrics_simpo

    # ORPO
    loss_orpo, metrics_orpo = orpo_loss(sft_loss=1.2, policy_chosen_logps=policy_chosen, policy_rejected_logps=policy_rejected)
    assert loss_orpo > 0.0
    assert "total_loss" in metrics_orpo

    # KTO
    loss_kto, metrics_kto = kto_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert loss_kto > 0.0
    assert "chosen_loss" in metrics_kto


def test_preference_data_extraction(tmp_path):
    """Verify preference dataset extraction for DPO/ORPO/SimPO."""
    dpo_file = tmp_path / "pref.jsonl"
    dpo_file.write_text(
        json.dumps({"prompt": "How to stay healthy?", "chosen": "Eat well and exercise.", "rejected": "Do nothing."}) + "\n" +
        json.dumps({"instruction": "Write poem", "chosen": "Roses are red.", "rejected": "No poem."}) + "\n"
    )
    pairs = extract_preference_pairs(dpo_file)
    assert len(pairs) == 2
    assert pairs[0]["prompt"] == "How to stay healthy?"
    assert pairs[0]["chosen"] == "Eat well and exercise."
    assert pairs[0]["rejected"] == "Do nothing."


def test_reward_synth_json_schema(tmp_path):
    """Verify JSON schema verifier synthesis and calibration."""
    json_refs = [
        '{"status": "ok", "count": 42}',
        '{"status": "success", "count": 100}',
        '{"status": "ready", "count": 5}',
    ]
    cand = infer_verifier(json_refs)
    assert cand.family == "json_schema"
    assert "status" in cand.parameters["required_keys"]
    assert "count" in cand.parameters["required_keys"]

    calib = calibrate_verifier(cand, json_refs)
    assert calib.reference_pass_rate == 1.0
    assert calib.negative_rejection_rate >= 0.80
    assert calib.verdict == "CALIBRATED"

    # End-to-end synthesis
    ref_file = tmp_path / "refs.jsonl"
    ref_file.write_text("\n".join(json.dumps({"output": r}) for r in json_refs))
    out_py = tmp_path / "reward.py"
    report_json = tmp_path / "calib.json"

    synthesize_reward_script(ref_file, out_py, report_json)
    assert out_py.exists()
    assert report_json.exists()
    assert "def verify" in out_py.read_text(encoding="utf-8")


def test_reward_synth_tool_calling(tmp_path):
    """Verify Tool Call verifier synthesis and calibration."""
    tool_refs = [
        '{"name": "fetch_weather", "arguments": {"city": "Berlin"}}',
        '{"name": "fetch_weather", "arguments": {"city": "Tokyo"}}',
    ]
    cand = infer_verifier(tool_refs)
    assert cand.family == "tool_call"
    assert "fetch_weather" in cand.parameters["expected_names"]

    calib = calibrate_verifier(cand, tool_refs)
    assert calib.verdict == "CALIBRATED"


def test_reward_synth_numeric():
    """Verify numeric verifier synthesis and calibration."""
    num_refs = ["The temperature is 98.6 degrees.", "Value: 99.1", "Score = 98.0"]
    cand = infer_verifier(num_refs)
    assert cand.family == "numeric"
    calib = calibrate_verifier(cand, num_refs)
    assert calib.verdict == "CALIBRATED"


def test_ship_gate_verdict(tmp_path):
    """Verify Leg-2 regression gate execution and SHIP verdict."""
    adapter_dir = tmp_path / "mock_adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.bin").write_text("DUMMY_WEIGHTS")

    evidence_dir = tmp_path / "evals"
    verdict = run_ship_gate(adapter_path=adapter_dir, output_evidence_dir=evidence_dir)

    assert verdict.verdict == "SHIP"
    assert verdict.exit_code == 0
    assert verdict.overall_score >= 0.75
    assert len(verdict.suites) == 4
    assert verdict.evidence_path.exists()

    evidence_data = json.loads(verdict.evidence_path.read_text(encoding="utf-8"))
    assert evidence_data["verdict"] == "SHIP"
    assert len(evidence_data["suites"]) == 4


def test_merge_adapter_and_gguf_export(tmp_path):
    """Verify LoRA merging and GGUF/Modelfile export."""
    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "mock"}')
    (adapter_dir / "tokenizer.json").write_text('{"mock": true}')

    # Merge
    merged_dir = tmp_path / "merged_output"
    out_dir = merge_adapter(base_dir, adapter_dir, merged_dir)
    assert out_dir.exists()
    assert (merged_dir / "config.json").exists()
    assert (merged_dir / "model.safetensors").exists()

    # GGUF Export & Modelfile
    gguf_file = tmp_path / "model-q4_k_m.gguf"
    export_to_gguf(merged_dir, gguf_file, quant="q4_k_m")
    assert gguf_file.exists()
    assert gguf_file.read_bytes()[:4] == b"GGUF"

    modelfile = tmp_path / f"Modelfile.{gguf_file.stem}"
    assert modelfile.exists()
    assert "FROM ./model-q4_k_m.gguf" in modelfile.read_text(encoding="utf-8")
