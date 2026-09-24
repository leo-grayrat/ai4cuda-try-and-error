"""Code-first optimization localization tests."""

from kernel_lab.code_credit import locate_actions, locate_edits, revert_action, revert_edit


def test_locates_kernel_change_before_boilerplate() -> None:
    parent = "def helper():\n    return 1\n\n@triton.jit\ndef kernel(x, y):\n    y = x\n"
    child = "def helper():\n    return 2\n\n@triton.jit\ndef kernel(x, y):\n    y = x + 1\n"
    units = locate_edits(parent, child, language="python")
    assert units[0]["category"] == "kernel_compute"
    assert units[0]["child_lines"] == [6, 6]
    assert any(unit["category"] == "boilerplate" for unit in units)


def test_locates_cuda_kernel_edit_above_host_binding() -> None:
    parent = (
        "__global__ void add(const float* x, float* y) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  y[i] = x[i];\n"
        "}\n"
        "int bind() { return 1; }\n"
    )
    child = parent.replace("y[i] = x[i];", "y[i] = x[i] + 1;").replace("return 1", "return 2")
    units = locate_edits(parent, child, language="cuda")
    assert units[0]["category"] == "kernel_compute"
    assert units[0]["child_lines"] == [3, 3]
    assert units[-1]["category"] == "boilerplate"


def test_global_tf32_switch_is_not_kernel_algorithm_edit() -> None:
    parent = "def forward(x):\n    return x.relu_()\n"
    child = "torch.backends.cuda.matmul.allow_tf32 = True\n" + parent
    units = locate_edits(parent, child, language="python")
    assert len(units) == 1
    assert units[0]["category"] == "runtime_config"


def test_revert_only_selected_edit() -> None:
    parent = "@triton.jit\ndef kernel(x, y):\n    y = x\n\ndef helper():\n    return 1\n"
    child = parent.replace("y = x", "y = x + 1").replace("return 1", "return 2")
    units = locate_edits(parent, child, language="python")
    reverted = revert_edit(parent, child, units[0])
    assert "y = x\n" in reverted
    assert "return 2\n" in reverted


def test_comment_only_change_does_not_become_optimization() -> None:
    parent = "@triton.jit\ndef kernel(x, y):\n    # load input\n    y = x\n"
    child = parent.replace("# load input", "# tl.load could improve performance")
    units = locate_edits(parent, child, language="python")
    assert units[0]["category"] == "boilerplate"


def test_kernel_docstring_change_is_not_an_optimization() -> None:
    parent = "@triton.jit\ndef kernel(x, y):\n    \"\"\"Writes results.\"\"\"\n    y = x\n"
    child = parent.replace("Writes results.", "Avoids atomics and tl.load.")
    units = locate_edits(parent, child, language="python")
    assert len(units) == 1
    assert units[0]["category"] == "boilerplate"


def test_docstring_cannot_turn_an_ordinary_function_into_a_kernel() -> None:
    parent = "def helper(x):\n    \"\"\"Could use tl.load someday.\"\"\"\n    return x + 1\n"
    child = parent.replace("return x + 1", "return x + 2")
    units = locate_edits(parent, child, language="python")
    assert units[0]["category"] == "boilerplate"


def test_docs_do_not_join_a_kernel_transformation() -> None:
    parent = "@triton.jit\ndef kernel(x, y):\n    \"\"\"Old notes.\"\"\"\n\n    y = x\n"
    child = parent.replace("Old notes.", "New notes.").replace("y = x\n", "y = x + 1\n")
    actions = locate_actions(parent, child, language="python")
    assert len(actions) == 2
    assert len(actions[0]["edits"]) == 1
    assert actions[0]["category"] == "kernel_compute"


def test_new_fast_path_condition_is_a_candidate() -> None:
    parent = "def forward(x):\n    return x + 1\n"
    child = "def forward(x):\n    if hasattr(x, 'fast_path'):\n        return x.fast_path()\n    return x + 1\n"
    units = locate_edits(parent, child, language="python")
    assert units[0]["category"] == "host_compute"


def test_long_unchanged_host_code_is_not_returned() -> None:
    prefix = "".join(f"def ordinary_{i}(): return {i}\n" for i in range(1000))
    parent = prefix + "@triton.jit\ndef kernel(x, y):\n    y = x\n"
    child = parent.replace("    y = x\n", "    y = x + 1\n")
    units = locate_edits(parent, child, language="python")
    assert len(units) == 1
    assert units[0]["category"] == "kernel_compute"


def test_device_copy_change_is_a_performance_candidate() -> None:
    parent = "def forward(x):\n    x = x.cuda()\n    return x\n"
    child = "def forward(x):\n    if not x.is_cuda: raise ValueError('device')\n    return x\n"
    units = locate_edits(parent, child, language="python")
    assert units[0]["category"] == "host_transfer"


def test_autocast_is_classified_as_precision_change() -> None:
    parent = "def forward(x):\n    return self.linear(x)\n"
    child = "def forward(x):\n    with torch.cuda.amp.autocast(dtype=torch.float16):\n        return self.linear(x)\n"
    units = locate_edits(parent, child, language="python")
    assert units[0]["category"] == "precision"


def test_related_kernel_and_launch_edits_form_one_action() -> None:
    parent = (
        "@triton.jit\ndef kernel(x, y, BLOCK: tl.constexpr):\n"
        "    i = tl.arange(0, BLOCK)\n    tl.store(y + i, tl.load(x + i))\n\n"
        "def forward(x, y):\n    kernel[(1,)](x, y, BLOCK=128)\n"
    )
    child = parent.replace("BLOCK: tl.constexpr", "BLOCK: tl.constexpr, ROW_BLOCK: tl.constexpr")
    child = child.replace("tl.arange(0, BLOCK)", "tl.arange(0, BLOCK * ROW_BLOCK)")
    child = child.replace("BLOCK=128)", "BLOCK=128, ROW_BLOCK=4)")
    actions = locate_actions(parent, child, language="python")
    assert len(actions) == 1
    assert len(actions[0]["edits"]) == 2
    assert revert_action(parent, child, actions[0]) == parent
