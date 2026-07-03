"""Quick smoke test of the DSL: build a few primitives and validate them."""
import sys
sys.path.insert(0, "/home/z/my-project")

from neurogolf import dsl, validator, arc_data


def test_identity():
    """Identity network should pass through any input unchanged."""
    print("=== Test 1: Identity ===")
    m = dsl.identity()
    # Pick a task that is identity-like — we'll just check it doesn't crash.
    task = arc_data.load_task(1)
    e = validator.evaluate_model(m, task)
    print(f"  Identity on task 1: eligible={e['eligible_for_points']}, score={e['score']:.2f}")
    print(f"  (Params={e['params']}, Size={e['size_bytes']} bytes)")


def test_color_map():
    """Test color_map on a known color-substitution task."""
    print("\n=== Test 2: Color map (1 -> 2) ===")
    m = dsl.color_map({1: 2})
    task = arc_data.load_task(1)
    e = validator.evaluate_model(m, task)
    print(f"  Color map {{1:2}} on task 1: eligible={e['eligible_for_points']}")


def test_conv_layer():
    """Test single_layer_conv2d on a task that needs local filtering."""
    print("\n=== Test 3: Single-layer conv2d (identity weights) ===")
    # Identity conv: 3x3 with weight 1 only at center, on diagonal
    W = dsl.conv_weight_from_fn(
        lambda co, ci, k: 1.0 if (k == (0, 0) and co == ci) else 0.0,
        kernel_size=3,
    )
    m = dsl.single_layer_conv2d(W)
    task = arc_data.load_task(1)
    e = validator.evaluate_model(m, task)
    print(f"  Identity conv on task 1: eligible={e['eligible_for_points']}, score={e['score']:.2f}")
    print(f"  Params={e['params']}, Size={e['size_bytes']} bytes")


def test_chain():
    """Test chaining two color maps: 1 -> 2 -> 3."""
    print("\n=== Test 4: Chain (color map 1->2) then (color map 2->3) ===")
    m1 = dsl.color_map({1: 2})
    m2 = dsl.color_map({2: 3})
    m = dsl.chain([m1, m2])
    task = arc_data.load_task(1)
    e = validator.evaluate_model(m, task)
    print(f"  Chained color maps on task 1: eligible={e['eligible_for_points']}, score={e['score']:.2f}")
    print(f"  Params={e['params']}, Size={e['size_bytes']} bytes")


def test_argmax_layer():
    """Test the argmax/one-hot final layer."""
    print("\n=== Test 5: Argmax+OneHot layer ===")
    m = dsl.argmax_over_channels()
    task = arc_data.load_task(1)
    e = validator.evaluate_model(m, task)
    print(f"  Argmax layer on task 1: eligible={e['eligible_for_points']}")
    s_ok, s_msg = dsl.validate_model_structure(m)
    print(f"  Structural: {s_msg}")


def test_constant_grid():
    """Test constant_grid output."""
    print("\n=== Test 6: Constant grid output ===")
    grid = [[1, 2], [3, 4]]
    m = dsl.constant_grid(grid)
    task = arc_data.load_task(1)
    # We just test that it runs and structural checks pass
    s_ok, s_msg = dsl.validate_model_structure(m)
    print(f"  Structural: {s_msg}")
    out = validator.run_model(m, [[0, 0], [0, 0]])
    pred = arc_data.onehot_to_grid(out, 2, 2)
    print(f"  Output: {pred} (expected {grid})")


if __name__ == "__main__":
    test_identity()
    test_color_map()
    test_conv_layer()
    test_chain()
    test_argmax_layer()
    test_constant_grid()
