import numpy as np

from experiments.train_mlp_xor import train_xor
from localml_scholar.losses import softmax_cross_entropy_loss_and_gradient
from localml_scholar.models.mlp import MLP
from localml_scholar.optim.adam import Adam


def test_mlp_learns_xor_deterministically() -> None:
    inputs = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    targets = np.array([0, 1, 1, 0], dtype=np.int64)
    model = MLP(2, 8, 2, activation="gelu", seed=17)
    optimizer = Adam(model.parameters(), learning_rate=0.05)

    model.eval()
    initial_loss, _ = softmax_cross_entropy_loss_and_gradient(
        model.forward(inputs), targets
    )
    model.train()
    for _ in range(500):
        optimizer.zero_grad()
        logits = model.forward(inputs)
        _, grad_logits = softmax_cross_entropy_loss_and_gradient(logits, targets)
        model.backward(grad_logits)
        optimizer.step()

    model.eval()
    final_logits = model.forward(inputs)
    final_loss, _ = softmax_cross_entropy_loss_and_gradient(final_logits, targets)
    predictions = np.argmax(final_logits, axis=-1)

    assert final_loss < 3e-5
    assert final_loss < initial_loss * 1e-3
    assert np.array_equal(predictions, targets)


def test_mlp_checkpoint_preserves_configuration_and_predictions(tmp_path) -> None:
    model = MLP(2, 5, 3, activation="relu", seed=9)
    inputs = np.array([[0.2, 0.8], [-0.5, 0.3]], dtype=np.float64)
    model.eval()
    expected = model.forward(inputs)
    checkpoint = tmp_path / "mlp.npz"

    model.save_checkpoint(checkpoint)
    loaded = MLP.load_checkpoint(checkpoint).eval()

    assert loaded.configuration == model.configuration
    assert loaded.parameter_count == model.parameter_count
    assert np.array_equal(loaded.forward(inputs), expected)


def test_xor_experiment_writes_verified_summary_and_checkpoint(tmp_path) -> None:
    summary = train_xor(
        seed=17,
        steps=500,
        learning_rate=0.05,
        hidden_dim=8,
        output_directory=tmp_path,
        report_interval=250,
    )

    assert summary["correct_predictions"] == 4
    assert summary["final_loss"] < 3e-5
    assert summary["checkpoint_round_trip_exact"]
    assert (tmp_path / "model.npz").is_file()
    assert (tmp_path / "optimizer.npz").is_file()
    assert (tmp_path / "run_summary.json").is_file()
